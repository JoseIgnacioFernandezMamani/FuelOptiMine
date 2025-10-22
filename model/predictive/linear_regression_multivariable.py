import polars as pl
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn import linear_model
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Any
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import datetime

from model.utils.model_utils import log_results, get_logger
from etl_core.load.utils.config import CH_CONFIG, create_client
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning


class LinearRegressionModel:
    """
    Clase simplificada para regresión lineal con dos modelos: Stage 4 y Stage 8
    """

    def __init__(self, truck_id: str):
        self.cycles_data: pl.DataFrame = pl.DataFrame()
        self.predictor_vars: list[str] = [
            "Distance",
            "CycleDurationSeconds",  # no incluyo tonage porque no aporta nada realmente a la prediccion.
        ]

        # Escaladores para cada stage
        self.scaler_stage_4: StandardScaler = StandardScaler()
        self.scaler_stage_8: StandardScaler = StandardScaler()

        # Modelos
        self.model_stage_4 = LinearRegression()
        self.model_stage_8 = LinearRegression()

        # modelo de isolation forest para detección de outliers
        self.iso_forest_stage_4: IsolationForest = IsolationForest(
            contamination=0.5, random_state=42
        )
        self.iso_forest_stage_8: IsolationForest = IsolationForest(
            contamination=0.5, random_state=42
        )

        # results
        self.results_stage_4: dict = {}
        self.results_stage_8: dict = {}

        self.df: pl.DataFrame = pl.DataFrame()

        # self.logger
        self.logger = get_logger("LinearRegression", "lrm.log", console=False)

        # self total consumed
        self.total_consumed: float = 0.0

        # stage 1 fuel consumed estimated
        self.st1_fuel_consumed: float = 0.0
        # truck id
        self.truck_id = truck_id

    def load_data(self):
        """
        Load data from ClickHouse or CSV file.

        Args:
            truck_id: ID del camión específico (ej: 'T-210'). Si None, carga todos los camiones
            from_clickhouse: Si True carga desde ClickHouse, si False desde CSV
        """
        self.logger.info(f"Cargando datos desde ClickHouse para {self.truck_id}")
        client = create_client(CH_CONFIG, self.logger)

        query = f"""
        SELECT *
        FROM {CH_CONFIG['database']}.xgboost_fuel
        WHERE Equipment = '{self.truck_id}'
        ORDER BY SortTimestamp
        """

        result = client.query(query)

        # Crear DataFrame de pandas manualmente
        columns = result.column_names
        data = result.result_rows

        df_pandas = pd.DataFrame(data, columns=columns)

        # Convertir a polars
        self.df = pl.from_pandas(df_pandas)

        # get total consumed fuel
        self.total_consumed = self.df.select(pl.col("DeltaFuel").sum()).item()

        # get stage 1 consumed fuel estimated

        client.close()

        self.logger.info(f"Datos cargados exitosamente: {len(self.df)} registros")

    @staticmethod
    def theil_fuel_consumption(fuel_levels: List[float]) -> float:
        """
        Función Theil-Sen para cálculo robusto de consumo de combustible en rangos de datos de nivel de combustible, son pocos datos, la idea es aproximar una tendencia a la baja para obtener una aproximacion del consumo de combustible.

        Args:
            fuel_levels: Lista de niveles de combustible ordenados temporalmente
        Returns:
            float: Consumo de combustible estimado (diferencia inicio-fin de la regresión)
        """
        if len(fuel_levels) < 3:
            return max(0.0, fuel_levels[0] - fuel_levels[-1])

        try:
            # Filtrar valores <= 0 y limpiar repetidos excesivos
            valid_fuel_levels = []
            consecutive_count = 0
            last_value = None

            for v in fuel_levels:
                if v > 0:  # Ignorar valores <= 0
                    if v == last_value:
                        consecutive_count += 1
                        # Solo mantener hasta 3 valores consecutivos repetidos
                        if consecutive_count <= 3:
                            valid_fuel_levels.append(v)
                    else:
                        consecutive_count = 1
                        last_value = v
                        valid_fuel_levels.append(v)
            y = np.array(valid_fuel_levels, dtype=np.float64)

            if len(y) < 3:
                return max(0.0, fuel_levels[0] - fuel_levels[-1])

            X = np.arange(1, len(y) + 1).reshape(-1, 1)

            # Theil-Sen Regressor - robust and stable
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                # Theil-Sen with optimized parameters for small datasets
                theil_sen = linear_model.TheilSenRegressor(
                    fit_intercept=True,
                    max_subpopulation=min(1e4, len(fuel_levels) * 100),
                    n_subsamples=min(len(fuel_levels), 50),
                    max_iter=100,
                    tol=1e-2,
                    random_state=42,
                )

                theil_sen.fit(X, y)

            # Calcular predicciones
            pred_inicio = theil_sen.predict([[1]])[0]
            pred_fin = theil_sen.predict([[len(y)]])[0]

            return abs(pred_inicio - pred_fin)

        except Exception:
            # Fallback al método simple
            return max(0.0, fuel_levels[0] - fuel_levels[-1])

    def transform_cycles_data(self):
        """
        Procesa datos para identificar ciclos y calcular métricas de consumo de combustible.

        Este método realiza una transformación integral de datos para identificar ciclos
        operacionales de camiones y calcular el consumo de combustible por ciclo.
        Procesa datos de sensores para crear agregaciones significativas basadas en ciclos
        para entrenamiento de modelos.

        La transformación incluye:
            - Identificación de ciclos basada en secuencias de etapas
            - Suavizado de nivel de combustible usando mediana móvil
            - Unificación de datos geográficos
            - Agrupación y agregación de ciclos
            - Cálculo de consumo de combustible
            - Filtrado de calidad de datos

        Los resultados se almacenan en self.cycles_data como DataFrame de Polars.
                Procesa datos para identificar bloques basados en transiciones de stages.

        Este método identifica 3 tipos de bloques:
        1. Stage 1->4: Desde StageSequence=1 hasta StageSequence=4 (ciclo de carga/descarga)
        2. Stage 4->8: Desde StageSequence=4 hasta StageSequence=8 (ciclo de retorno/posicionamiento)
        3. Stage 8->1: Desde StageSequence=8 hasta StageSequence=1 (tiempo libre/sin ciclo)

        Los timestamps se extraen de los registros donde están presentes los stages específicos.
        """
        if self.df is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        self.logger.info("Iniciando transformación de datos de ciclos...")

        df = self.df.with_columns(
            [
                # rolling median del nivel de combustible
                pl.col("FuelLevelLiters")
                # .rolling_median(window_size=10, min_samples=3, center=True)
                # .fill_null(0)
                .alias("MedianFuelLevelLiters"),
                # unificar destinos
                pl.coalesce([pl.col("LoadingZone"), pl.col("Destination")]).alias(
                    "Destination"
                ),
                # datos geográficos
                pl.when(pl.col("Latitude_cycle") != 0)
                .then(pl.col("Latitude_cycle"))
                .otherwise(pl.col("Latitude"))
                .alias("Latitude"),
                pl.when(pl.col("Longitude_cycle") != 0)
                .then(pl.col("Longitude_cycle"))
                .otherwise(pl.col("Longitude"))
                .alias("Longitude"),
                pl.when(pl.col("Elevation_cycle") != 0)
                .then(pl.col("Elevation_cycle"))
                .otherwise(pl.col("Elevation"))
                .alias("Elevation"),
                # para obtener la mediana aproximada
                pl.col("FuelLevelLiters").diff().abs().alias("delta_fuel"),
            ]
        )

        df = df.with_columns(
            # identificar ciclos
            pl.when(
                (pl.col("StageSequence") == 4)
                | (pl.col("StageSequence") == 8)
                | (pl.col("StageSequence") == 1)
            )
            .then(True)
            .otherwise(False)
            .alias("cycle_end")
        )

        # crear grupos de ciclos, se obtuvo los registros enteros de cada grupo
        df = df.with_columns(
            [
                pl.col("cycle_end")
                .shift(1, fill_value=False)
                .cum_sum()
                .alias("cycle_group")
            ]
        )

        result = (
            df.group_by("cycle_group")
            .agg(
                [
                    # metadatos
                    pl.len().alias(
                        "RecordsInCycle"
                    ),  # numero de registros de sensor, metadato
                    pl.col("StageSequence").sum().alias("StageSum"),  # columna temporal
                    pl.col("TimeStampIni").last().alias("TimeStampIni"),
                    pl.col("TimeStampFin").last().alias("TimeStampFin"),
                    pl.col("ShiftDate").last().alias("ShiftDate"),
                    pl.col("Shift").last().alias("Shift"),
                    pl.col("Equipment").last().alias("Equipment"),
                    pl.col("TruckFleet")
                    .last()
                    .alias("TruckFleet"),  # fin de los metadatos
                    # variables para calculo, consumo de combustible
                    pl.col("MedianFuelLevelLiters").alias("FuelLevelsList"),
                    # variables numericas para modelo
                    (
                        pl.col("SpeedAvg")
                        .filter(pl.col("SpeedAvg") > 5)
                        .median()
                        .fill_null(pl.col("SpeedAvg").median())
                    ).alias("SpeedAvg"),
                    pl.col("SlopePercent").median().alias("AvgSlopePercent"),
                    pl.col("Acceleration").median().alias("AvgAcceleration"),
                    pl.col("TimeEfficiencyPercentage")
                    .sum()
                    .alias("TimeEfficiencyPercentage"),
                    pl.col("Latitude").last().alias("Latitude"),
                    pl.col("Longitude").last().alias("Longitude"),
                    pl.col("Elevation").last().alias("Elevation"),
                    # variables categoricas
                    pl.col("StageSequence").last().alias("StageSequence"),
                    pl.col("Destination").last().alias("Destination"),
                    pl.col("DestinationType").last().alias("DestinationType"),
                    pl.col("Material").last().alias("Material"),
                    pl.col("Shovel").last().alias("Shovel"),
                    # datos reales de los ciclos
                    pl.col("MeasuredTonnage").sum().alias("TotalMeasuredTonnage"),
                    pl.col("Distance").sum().alias("Distance"),
                ]
            )
            .sort("cycle_group")
        )

        result = result.filter(
            (pl.col("StageSum") == 9) & (pl.col("StageSequence") == 4)
            | (pl.col("StageSum") == 26) & (pl.col("StageSequence") == 8)
        ).drop("StageSum")

        # obtener los verdaderos timestamps de inicio y fin del ciclo
        result = result.with_columns(
            # Dynamic timestamp selection initial
            pl.when(
                (pl.col("StageSequence") == 4) & (pl.col("StageSequence").shift(1) == 1)
            )
            .then(pl.col("TimeStampIni").shift(1))
            .when(
                (pl.col("StageSequence") == 8) & (pl.col("StageSequence").shift(1) == 4)
            )
            .then(pl.col("TimeStampFin").shift(1))
            .when(
                (pl.col("StageSequence") == 1) & (pl.col("StageSequence").shift(1) == 8)
            )
            .then(pl.col("TimeStampFin").shift(1))
            .otherwise(pl.col("TimeStampIni"))
            .alias("TimeStampIni"),
            # dynamic timestamp selection final
            pl.when(
                (pl.col("StageSequence") == 1) & (pl.col("StageSequence").shift(1) == 8)
            )
            .then(pl.col("TimeStampIni"))
            .otherwise(pl.col("TimeStampFin"))
            .alias("TimeStampFin"),
            # Combustible consumido (sin cambios)
            # pl.col("MedianFuelLevelLiters").diff().abs().alias("FuelConsumed"),
            # ajustar TimeEfficiencyPercentage
            pl.when(
                (pl.col("StageSequence") == 4)
                & (pl.col("StageSequence").shift(1) == 1)
                & (pl.col("TimeEfficiencyPercentage").shift(1) > 0)
            )
            .then(
                pl.col("TimeEfficiencyPercentage").shift(1)
                + pl.col("TimeEfficiencyPercentage")
            )
            .otherwise(pl.col("TimeEfficiencyPercentage"))
            .alias("TimeEfficiencyPercentage"),
        )

        result = result.with_columns(
            pl.when(
                (
                    (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                    .dt.total_seconds()
                    .abs()
                    > 3600
                )
                | (
                    (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                    .dt.total_seconds()
                    .abs()
                    < 50
                )
                & (pl.col("SpeedAvg") > 0.1)
                & (pl.col("Distance") > 50)
                & (pl.col("Distance") <= 3600)
            )
            .then(
                # Duración = distancia(m) / velocidad(m/s)
                pl.min_horizontal(
                    [
                        (pl.col("Distance") / (pl.col("SpeedAvg") / 3.6)),
                        pl.lit(float(900)),
                    ]
                ).clip(lower_bound=180)
            )
            .otherwise(
                (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                .dt.total_seconds()
                .abs()
            )
            .alias("CycleDurationSeconds"),
            pl.when(pl.col("StageSequence") == 1)
            .then(0)
            .otherwise(pl.col("TimeEfficiencyPercentage"))
            .alias("TimeEfficiencyPercentage"),
            # Destino con limpieza mejorada
            pl.when(pl.col("Destination").str.strip_chars().str.len_bytes() > 2)
            .then(pl.col("Destination"))
            .otherwise(pl.lit("UNKNOWN"))
            .alias("Destination"),
            # Tipo de destino con valor por defecto
            pl.col("DestinationType")
            .fill_null(pl.lit("LoadingZone"))
            .alias("DestinationType"),
            # Material con valor por defecto
            pl.col("Material").fill_null(pl.lit("Empty")).alias("Material"),
            pl.col("Shovel").fill_null(pl.lit("Unknown")).alias("Shovel"),
        )

        # calcular el consumo de stage 1
        st1 = (
            result.filter(pl.col("StageSequence") == 1)
            .with_columns(
                pl.col("FuelLevelsList").list.first().alias("StartFuel"),
                pl.col("FuelLevelsList").list.last().alias("EndFuel"),
            )
            .with_columns(
                pl.when(
                    (pl.col("StartFuel") - pl.col("EndFuel") < 30)
                    & (pl.col("StartFuel") - pl.col("EndFuel") >= 0)
                )
                .then(pl.col("StartFuel") - pl.col("EndFuel"))
                .otherwise(0)
                .alias("aux")
            )
        )
        self.st1_fuel_consumed = st1.select(pl.col("aux")).sum().item()

        # solo obtener ciclos de carguio 4 y acarreo 8
        result = result.filter(
            (
                (pl.col("StageSequence").is_in([4, 8]))
                & (pl.col("StageSequence").is_not_null())
            )
        )

        # calcular combustible consumido usando theil simplificado
        result = result.with_columns(
            [
                pl.col("FuelLevelsList")
                .map_elements(
                    lambda fuel_list: LinearRegressionModel.theil_fuel_consumption(
                        fuel_list
                    ),
                    return_dtype=pl.Float64,
                )
                .alias("FuelConsumed")
            ]
        )

        # eliminar fuellevel list
        result = result.drop("FuelLevelsList")
        self.cycles_data = self.best_data_for_train(result)

        self.logger.info(
            f"Transformación completada: {len(self.cycles_data)} ciclos procesados"
        )

    def best_data_for_train(self, result: pl.DataFrame) -> pl.DataFrame:

        # obtener dataframes separados por stage
        df_stage4 = result.filter(pl.col("StageSequence") == 4)
        df_stage8 = result.filter(pl.col("StageSequence") == 8)

        stats = {"stage4": {}, "stage8": {}}
        # columnas específicas por stage
        cols_stage4 = ["Distance", "CycleDurationSeconds", "FuelConsumed"]
        cols_stage8 = [
            "TotalMeasuredTonnage",
            "Distance",
            "CycleDurationSeconds",
            "FuelConsumed",
        ]

        # --- Stage 4 ---
        for col in cols_stage4:
            values = df_stage4.select(
                [
                    pl.col(col).quantile(0.10).alias("q1"),
                    pl.col(col).quantile(0.90).alias("q3"),
                ]
            ).row(0)

            q1, q3 = values

            stats["stage4"][col] = {
                "q1": q1,
                "q3": q3,
            }

        # --- Stage 8 ---
        for col in cols_stage8:
            values = df_stage8.select(
                [
                    pl.col(col).quantile(0.1).alias("q1"),
                    pl.col(col).quantile(0.90).alias("q3"),
                ]
            ).row(0)

            q1, q3 = values

            stats["stage8"][col] = {
                "q1": q1,
                "q3": q3,
            }

        # ordenar por fuelConsumed
        df_stage4 = df_stage4.sort("FuelConsumed")
        df_stage4 = df_stage4.with_columns(
            pl.col("Distance").diff().alias("delta_distance"),
            pl.col("CycleDurationSeconds").diff().alias("delta_duration"),
            pl.col("TotalMeasuredTonnage").diff().alias("delta_tonnage"),
        )
        df_stage8 = df_stage8.sort("FuelConsumed")
        df_stage8 = df_stage8.with_columns(
            pl.col("Distance").diff().alias("delta_distance"),
            pl.col("CycleDurationSeconds").diff().alias("delta_duration"),
            pl.col("TotalMeasuredTonnage").diff().alias("delta_tonnage"),
        )

        # aplicar filtro de buen dato o mal dato
        df_stage4 = df_stage4.with_columns(
            pl.when(
                (pl.col("FuelConsumed") <= stats["stage4"]["FuelConsumed"]["q1"])
                | (pl.col("FuelConsumed") >= stats["stage4"]["FuelConsumed"]["q3"])
                | (pl.col("Distance") <= stats["stage4"]["Distance"]["q1"])
                | (pl.col("Distance") >= stats["stage4"]["Distance"]["q3"])
                | (
                    pl.col("CycleDurationSeconds")
                    <= stats["stage4"]["CycleDurationSeconds"]["q1"]
                )
                | (
                    pl.col("CycleDurationSeconds")
                    >= stats["stage4"]["CycleDurationSeconds"]["q3"]
                )
                | (
                    # Regla de patrón de crecimiento: al menos uno debe crecer
                    ~((pl.col("delta_distance") > 0) | (pl.col("delta_duration") > 0))
                )
            )
            .then(False)
            .otherwise(True)
            .alias("QualityData")
        )

        df_stage8 = df_stage8.with_columns(
            pl.when(
                (pl.col("FuelConsumed") <= stats["stage8"]["FuelConsumed"]["q1"])
                | (pl.col("FuelConsumed") >= stats["stage8"]["FuelConsumed"]["q3"])
                | (pl.col("Distance") <= stats["stage8"]["Distance"]["q1"])
                | (pl.col("Distance") >= stats["stage8"]["Distance"]["q3"])
                | (
                    pl.col("CycleDurationSeconds")
                    <= stats["stage8"]["CycleDurationSeconds"]["q1"]
                )
                | (
                    pl.col("CycleDurationSeconds")
                    >= stats["stage8"]["CycleDurationSeconds"]["q3"]
                )
                | (
                    pl.col("TotalMeasuredTonnage")
                    <= stats["stage8"]["TotalMeasuredTonnage"]["q1"]
                )
                | (
                    pl.col("TotalMeasuredTonnage")
                    >= stats["stage8"]["TotalMeasuredTonnage"]["q3"]
                )
                | (
                    # Regla de patrón de crecimiento: al menos uno debe crecer
                    ~(
                        (pl.col("delta_distance") > 0)
                        | (pl.col("delta_duration") > 0)
                        | (pl.col("delta_tonnage") > 0)
                    )
                )
            )
            .then(False)
            .otherwise(True)
            .alias("QualityData")
        )

        concat_df = (
            pl.concat([df_stage4, df_stage8])
            .sort("cycle_group")
            .drop(["delta_distance", "delta_duration", "delta_tonnage"])
        )

        return concat_df

    def prepare_data_for_stage(
        self,
        data: pl.DataFrame,
        scaler: StandardScaler,
        iso_forest: IsolationForest,
        test_size: float = 0.2,
        random_state: int = 42,
        predictor_vars: List[str] = ["Distance", "CycleDurationSeconds"],
    ) -> Dict[str, Any]:
        """
        This method prepares cycle data by extracting features,
        splitting into train/test sets, and applying standardization scaling.

        Args:
            data (pl.DataFrame): Cycle data filtered for specific stage
            scaler (StandardScaler): Sklearn scaler for feature normalization
            test_size (float, optional): Proportion for test split. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducible splits. Defaults to 42.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - X_train: Scaled training features
                - X_test: Scaled test features
                - y_train: Training target values (fuel consumption)
                - y_test: Test target values (fuel consumption
        """
        self.logger.info(
            f"Preparando datos para stage con {len(data)} muestras iniciales"
        )

        # select predictor variables and target

        X = data.select(predictor_vars).to_pandas()
        y = data.select("FuelConsumed").to_numpy().flatten()

        self.logger.info(f"Variables predictoras: {predictor_vars}")

        # delete outliers using IsolationForest
        self.logger.info("Detectando outliers con Isolation Forest...")
        outlier_predictions = iso_forest.fit_predict(X)

        # Create boolean mask for inliers (1 = inlier, -1 = outlier)
        inlier_mask = outlier_predictions == 1
        n_outliers = np.sum(~inlier_mask)
        outlier_percentage = (n_outliers / len(X)) * 100

        self.logger.info(
            f"Outliers detectados: {n_outliers} ({outlier_percentage:.2f}%)"
        )

        # filter to keep only inliers
        X_clean = X[inlier_mask]
        y_clean = y[inlier_mask]

        self.logger.info(
            f"Datos después de remover outliers - X: {X_clean.shape}, y: {y_clean.shape}"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean,
            y_clean,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Prepare outlier information for logging
        outlier_info = {
            "n_outliers_detected": n_outliers,
            "outlier_percentage": outlier_percentage,
            "original_samples": len(X),
            "clean_samples": len(X_clean),
            "contamination_used": iso_forest.contamination,
        }

        self.logger.info(
            f"División final - Train: {len(X_train_scaled)}, Test: {len(X_test_scaled)}"
        )

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "outlier_info": outlier_info,
        }

    def calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calculate comprehensive regression performance metrics

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dict[str, float]: Dictionary containing performance metrics:
                - r2: R-squared score
                - mae: Mean Absolute Error
                - rmse: Root Mean Square Error
                - medae: Median Absolute Error
                - explained_variance: Explained Variance Score
                - mape: Mean Absolute Percentage Error
                - rmsle: Root Mean Square Log Error
        """
        metrics = {}
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[finite_mask], y_pred[finite_mask]

        if len(y_true) < 2:
            return {
                m: np.nan
                for m in [
                    "r2",
                    "mae",
                    "rmse",
                    "medae",
                    "explained_variance",
                    "mape",
                    "rmsle",
                ]
            }

        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["medae"] = median_absolute_error(y_true, y_pred)
        metrics["explained_variance"] = explained_variance_score(y_true, y_pred)
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        y_true_pos, y_pred_pos = np.maximum(y_true, 1e-10), np.maximum(y_pred, 1e-10)
        metrics["rmsle"] = np.sqrt(
            mean_squared_error(np.log1p(y_true_pos), np.log1p(y_pred_pos))
        )
        return metrics

    def train_models(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train both models without validation (Stage 4 and Stage 8)

        This method performs end-to-end training of regression models for two different
        manufacturing stages, including data preparation, model training, and evaluation.

        Args:
            test_size (float, optional): Proportion of dataset to include in the test split.
                                    Defaults to 0.2 (20%).
            random_state (int, optional): Random seed for reproducible train/test splits.
                                        Defaults to 42.

        Returns:
            Dict[str, Any]: Comprehensive training results containing:
                - stage_4: Dictionary with Stage 4 model results
                - stage_8: Dictionary with Stage 8 model results
                - multicollinearity: Multicollinearity analysis results

                Each stage dictionary contains:
                    - model: Trained RANSAC regression model
                    - scaler: Data scaler used for normalization
                    - samples: Train/test sample counts
                    - train_metrics: Training performance metrics
                    - test_metrics: Testing performance metrics
                    - model_parameters: Inverse-scaled model parameters
        """
        # load and transform data
        self.load_data()
        self.transform_cycles_data()

        # Separate data by stage for independent model training
        stage_4_data = self.cycles_data.filter(
            (pl.col("StageSequence") == 4) & pl.col("QualityData")
        )
        stage_8_data = self.cycles_data.filter(
            (pl.col("StageSequence") == 8) & pl.col("QualityData")
        )

        self.logger.info("Entrenando el modelo stage_4")

        # Prepare and scale Stage 4 data for training
        stage_4_scaled_data = self.prepare_data_for_stage(
            stage_4_data,
            self.scaler_stage_4,
            self.iso_forest_stage_4,
            test_size,
            random_state,
            self.predictor_vars,
        )
        if len(stage_4_scaled_data["y_train"]) < 3:
            self.logger.error(
                "No hay suficientes datos para entrenar el modelo Stage 4"
            )
            raise ValueError("No hay suficientes datos para entrenar el modelo Stage 4")

        self.model_stage_4 = LinearRegression().fit(
            stage_4_scaled_data["X_train"], stage_4_scaled_data["y_train"]
        )

        y_pred_train_4 = self.model_stage_4.predict(stage_4_scaled_data["X_train"])
        y_pred_test_4 = self.model_stage_4.predict(stage_4_scaled_data["X_test"])

        self.logger.info("Entrenando el modelo stage_8")

        # Prepare and scale Stage 8 data for training
        stage_8_scaled_data = self.prepare_data_for_stage(
            stage_8_data,
            self.scaler_stage_8,
            self.iso_forest_stage_8,
            test_size,
            random_state,
            self.predictor_vars,
        )
        if len(stage_8_scaled_data["y_train"]) < 3:
            self.logger.error(
                "No hay suficientes datos para entrenar el modelo Stage 8"
            )
            raise ValueError("No hay suficientes datos para entrenar el modelo Stage 8")

        self.model_stage_8 = LinearRegression().fit(
            stage_8_scaled_data["X_train"], stage_8_scaled_data["y_train"]
        )
        y_pred_train_8 = self.model_stage_8.predict(stage_8_scaled_data["X_train"])
        y_pred_test_8 = self.model_stage_8.predict(stage_8_scaled_data["X_test"])

        self.logger.info("Entrenamiento completado.")

        # Compile comprehensive results dictionary with all training outcomes
        result = {
            "stage_4": {
                "model": self.model_stage_4,
                "scaler": self.scaler_stage_4,
                "isolation_forest": self.iso_forest_stage_4,
                "samples": {
                    "train": len(stage_4_scaled_data["y_train"]),
                    "test": len(stage_4_scaled_data["y_test"]),
                },
                "train_metrics": self.calculate_metrics(
                    stage_4_scaled_data["y_train"], y_pred_train_4
                ),
                "test_metrics": self.calculate_metrics(
                    stage_4_scaled_data["y_test"], y_pred_test_4
                ),
                "outlier_info": stage_4_scaled_data["outlier_info"],
            },
            "stage_8": {
                "model": self.model_stage_8,
                "scaler": self.scaler_stage_8,
                "isolation_forest": self.iso_forest_stage_8,
                "samples": {
                    "train": len(stage_8_scaled_data["y_train"]),
                    "test": len(stage_8_scaled_data["y_test"]),
                },
                "train_metrics": self.calculate_metrics(
                    stage_8_scaled_data["y_train"], y_pred_train_8
                ),
                "test_metrics": self.calculate_metrics(
                    stage_8_scaled_data["y_test"], y_pred_test_8
                ),
                "outlier_info": stage_8_scaled_data["outlier_info"],
            },
            "total_consumed_fuel": self.total_consumed,
            "stage 1 fuel consumed": self.st1_fuel_consumed,
        }

        log_results(self.predictor_vars, "all", result, self.logger)

        return result

    def inverse_scale(self, model, scaler) -> Dict[str, Any]:
        """
        Extract model parameters in original units (undoing the scaling).
        """
        if not hasattr(model, "estimator_"):
            return {"error": "Modelo base no encontrado en RANSAC."}

        betas_scaled = model.estimator_.coef_
        intercept_scaled = model.estimator_.intercept_

        scale = scaler.scale_
        mean = scaler.mean_

        # Ajustar coeficientes a espacio original
        betas_original = betas_scaled / scale
        intercept_original = intercept_scaled - np.sum(betas_scaled * mean / scale)

        return {
            "intercept": intercept_original,
            "coefficients": dict(zip(self.predictor_vars, betas_original)),
        }

    def get_predictions(self) -> pl.DataFrame:
        """
        Generate fuel consumption predictions for all cycle data

        This method applies the trained RANSAC regression models to predict fuel consumption
        for all cycles in the dataset. It maintains the original data structure and order
        by processing each stage separately and then combining results ordered by timestamp.

        Returns:
            pl.DataFrame: Complete cycle data with added prediction column:
                - All original columns preserved in their original order
                - New column 'PredictedFuelConsumption' with model predictions
                - Data ordered by TimeStampIni to maintain chronological sequence

        Raises:
            ValueError: If models haven't been trained yet or cycle data is not available
            RuntimeError: If prediction fails due to data incompatibility
        """

        if self.cycles_data is None:
            raise ValueError("Cycle data not available. Run train_models() first.")
        if self.model_stage_4 is None or self.model_stage_8 is None:
            raise ValueError("Models not trained. Run train_models() first.")

        self.logger.info("Generating predictions for Stage 4 and Stage 8...")

        predictions = []

        # Stage 4
        stage_4_data = self.cycles_data.filter(pl.col("StageSequence") == 4)
        if len(stage_4_data) > 0:
            features = stage_4_data.select(self.predictor_vars).to_pandas()
            scaled = self.scaler_stage_4.transform(features)
            preds = self.model_stage_4.predict(scaled)

            stage_4_pred = stage_4_data.with_columns(
                [
                    pl.Series("PredictedFuelConsumption", preds),
                    pl.lit("stage_4").alias("ModelUsed"),
                ]
            )

            predictions.append(stage_4_pred)

            self.logger.info(f"Generated {len(preds)} predictions for Stage 4")

        # Stage 8
        stage_8_data = self.cycles_data.filter(pl.col("StageSequence") == 8)
        if len(stage_8_data) > 0:
            features = stage_8_data.select(self.predictor_vars).to_pandas()
            scaled = self.scaler_stage_8.transform(features)
            preds = self.model_stage_8.predict(scaled)

            stage_8_pred = stage_8_data.with_columns(
                [
                    pl.Series("PredictedFuelConsumption", preds),
                    pl.lit("stage_8").alias("ModelUsed"),
                ]
            )
            predictions.append(stage_8_pred)

            self.logger.info(f"Generated {len(preds)} predictions for Stage 8")

        if not predictions:
            raise RuntimeError("No data available for any model predictions.")

        # Concatenar y ordenar por TimeStampIni eliminar columna inecesaria
        predictions_df = pl.concat(predictions).sort("TimeStampIni")

        self.logger.info(f"Final predictions dataframe shape: {predictions_df.shape}")

        return predictions_df


if __name__ == "__main__":
    model = LinearRegressionModel()
    results = model.train_models()
    predictions_df = model.get_predictions()
    predictions_df.write_csv("predicted_cycles.csv")
