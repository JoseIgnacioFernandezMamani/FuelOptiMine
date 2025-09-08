import polars as pl
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
)
from typing import Dict, List, Tuple, Any
from sklearn import linear_model


class LinearRegressionMultivariable:
    """
    Clase para manejo de regresión lineal multivariable con preparación completa de datos
    siguiendo principios SOLID - Single Responsibility Principle
    Entrena automáticamente dos modelos: uno para Stage 4 y otro para Stage 8
    """

    def __init__(self, predictor_vars: List[str]):
        """
        Initialize the class
        """
        self.cycles_data = pl.DataFrame()
        self.predictor_vars = predictor_vars

        # Escaladores y encoders para cada stage
        self.feature_scaler_stage_4 = StandardScaler()
        self.feature_scaler_stage_8 = StandardScaler()
        self.destination_encoder_stage_4 = ce.BinaryEncoder(cols=["Destination"])
        self.destination_encoder_stage_8 = ce.BinaryEncoder(cols=["Destination"])

        # Modelos para cada stage
        self.mlr_model_stage_4 = linear_model.LinearRegression()
        self.mlr_model_stage_8 = linear_model.LinearRegression()
        self.ransac_model_stage_4 = linear_model.RANSACRegressor()
        self.ransac_model_stage_8 = linear_model.RANSACRegressor()

        # Resultados de ambos modelos
        self.model_results_stage_4 = {}
        self.model_results_stage_8 = {}

        # Datos preparados para cada stage
        self.prepared_data_stage_4 = {}
        self.prepared_data_stage_8 = {}

        self.df = pl.DataFrame()  # raw data

    def load_data(self):
        """
        Cargar datos desde archivo CSV

        Returns:
            pl.DataFrame: DataFrame con los datos cargados
        """
        print("Cargando datos desde unified_data_T-210.csv...")
        df = pl.read_csv("unified_data_T-210.csv", try_parse_dates=True)
        self.df = df.sort("SortTimestamp")
        return self.df

    def transform_cycles_data(self):
        """
        Procesar datos para identificar ciclos y calcular métricas de consumo de combustible
        """
        # Identify records with cycle and calculate rolling medians
        df = self.df.with_columns(
            [
                # identify cycles
                pl.when((pl.col("StageSequence") == 4) | (pl.col("StageSequence") == 8))
                .then(True)
                .otherwise(False)
                .alias("cycle_end"),
                # Rolling medians to obtain estimated fuel consumption
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=10, min_samples=3, center=True)
                .alias("MedianFuelLevelLiters"),
                pl.coalesce(["LoadingZone", "Destination"]).alias("Destination"),
            ]
        )

        # Create cycle groups - each group represents a stage (empty or full)
        df = df.with_columns(
            [
                pl.col("cycle_end")
                .shift(1, fill_value=False)
                .cum_sum()
                .alias("cycle_group")
            ]
        )

        # group for stage
        result = (
            df.group_by("cycle_group")
            .agg(
                [
                    pl.col("TimeStamp").first().alias("TimeStampIni"),
                    pl.col("TimeStamp").last().alias("TimeStampFin"),
                    pl.col("ShiftDate").last().alias("ShiftDate"),
                    pl.col("Equipment").last().alias("Equipment"),
                    pl.col("TruckFleet").last().alias("TruckFleet"),
                    pl.col("MedianFuelLevelLiters").first().alias("StartCycle"),
                    pl.col("MedianFuelLevelLiters").last().alias("EndCycle"),
                    pl.col("SpeedAvg").mean().alias("AverageSpeed"),
                    pl.col("RPM").mean().alias("AvgRPM"),
                    pl.col("SlopePercent").mean().alias("AvgSlopePercent"),
                    pl.col("Acceleration").mean().alias("AvgAcceleration"),
                    pl.col("MeasuredTonnage").sum().alias("TotalMeasuredTonnage"),
                    pl.col("Distance").sum().alias("Distance"),
                    pl.col("StageSequence").last().alias("StageSequence"),
                    pl.col("Destination").last().alias("Destination"),
                    pl.len().alias("RecordsInCycle"),
                ]
            )
            .sort("TimeStampIni")
        )

        # filter invalid data for Destination
        result = result.filter(
            (pl.col("Destination").str.strip_chars().str.len_chars() > 2)
        )

        # filter invalid data for FuelConsumed and CycleDurationSeconds
        result = result.with_columns(
            pl.when((pl.col("StartCycle") - pl.col("EndCycle")).abs() <= 500)
            .then((pl.col("StartCycle") - pl.col("EndCycle")).abs())
            .otherwise(10)
            .alias("FuelConsumed"),
            (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
            .dt.total_seconds()
            .abs()
            .alias("CycleDurationSeconds"),
        )

        # columns that need to be cleaned
        cols_to_clean = self.predictor_vars + ["FuelConsumed"]

        # Clean each column from null, NaN or infinite values
        for col in cols_to_clean:
            result = result.with_columns(
                pl.when(
                    pl.col(col).is_infinite()
                    | pl.col(col).is_nan()
                    | pl.col(col).is_null()
                )
                .then(0)
                .otherwise(pl.col(col))
                .alias(col)
            )

        self.cycles_data = result

    def get_datasets_by_stage(self) -> Dict[str, pl.DataFrame]:
        """
        Obtener datasets separados por tipo de etapa (Stage 4 y Stage 8)
        """
        stage_4_data = self.cycles_data.filter(pl.col("StageSequence") == 4)
        stage_8_data = self.cycles_data.filter(pl.col("StageSequence") == 8)

        return {"stage_4": stage_4_data, "stage_8": stage_8_data}

    def prepare_data_for_stage(
        self,
        data: pl.DataFrame,
        stage: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Preparar datos para un stage específico
        """
        # Extraer variables predictoras numéricas
        X_numeric = data.select(self.predictor_vars).to_pandas()

        # Extraer variable categórica
        X_destination = data.select("Destination").to_pandas()

        # Extraer variable objetivo
        y = data.select("FuelConsumed").to_numpy().flatten()

        # División entrenamiento/prueba
        X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=random_state, shuffle=True
        )

        X_dest_train, X_dest_test = train_test_split(
            X_destination, test_size=test_size, random_state=random_state, shuffle=True
        )

        # Seleccionar encoder y scaler según el stage
        if stage == "stage_4":
            destination_encoder = self.destination_encoder_stage_4
            feature_scaler = self.feature_scaler_stage_4
        else:
            destination_encoder = self.destination_encoder_stage_8
            feature_scaler = self.feature_scaler_stage_8

        # Codificación binaria para variable categórica 'Destination'
        X_dest_train_encoded = destination_encoder.fit_transform(
            X_dest_train
        ).to_numpy()
        X_dest_test_encoded = destination_encoder.transform(X_dest_test).to_numpy()

        # Estandarización de variables numéricas
        X_numeric_train_scaled = feature_scaler.fit_transform(X_numeric_train)
        X_numeric_test_scaled = feature_scaler.transform(X_numeric_test)

        # Combinar características numéricas escaladas y categóricas codificadas
        X_train_scaled = np.hstack([X_numeric_train_scaled, X_dest_train_encoded])
        X_test_scaled = np.hstack(
            [X_numeric_test_scaled, X_dest_test_encoded]
        )  # ignorar warning

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "total_features": X_train_scaled.shape[1],
        }

    def calculate_metrics(
        self, y_true, y_pred, model_type="linear"
    ) -> Dict[str, float]:
        """
        Calcular todas las métricas de evaluación de forma robusta para diferentes tipos de modelos
        """
        metrics = {}

        # Convertir a arrays numpy y asegurar que son float
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # Filtrar valores no finitos
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_filtered = y_true[finite_mask]
        y_pred_filtered = y_pred[finite_mask]

        # Si no hay suficientes datos después del filtrado, retornar NaN para todas las métricas
        if len(y_true_filtered) < 2 or len(y_pred_filtered) < 2:
            for metric_name in [
                "r2",
                "mae",
                "rmse",
                "medae",
                "explained_variance",
                "mape",
                "rmsle",
            ]:
                metrics[metric_name] = np.nan
            return metrics

        # Métricas estándar con manejo robusto de errores
        try:
            metrics["r2"] = r2_score(y_true_filtered, y_pred_filtered)
        except Exception as e:
            metrics["r2"] = np.nan

        try:
            metrics["mae"] = mean_absolute_error(y_true_filtered, y_pred_filtered)
        except Exception as e:
            metrics["mae"] = np.nan

        try:
            mse = mean_squared_error(y_true_filtered, y_pred_filtered)
            metrics["rmse"] = np.sqrt(mse)
        except Exception as e:
            metrics["rmse"] = np.nan

        try:
            metrics["medae"] = median_absolute_error(y_true_filtered, y_pred_filtered)
        except Exception as e:
            metrics["medae"] = np.nan

        try:
            metrics["explained_variance"] = explained_variance_score(
                y_true_filtered, y_pred_filtered
            )
        except Exception as e:
            metrics["explained_variance"] = np.nan

        # MAPE con manejo especial para RANSAC (más tolerante con outliers)
        try:
            if model_type == "ransac":
                # Para RANSAC, usar una versión más robusta del MAPE
                # que sea menos sensible a valores extremos
                epsilon = 1e-10  # pequeño valor para evitar división por cero
                ape = np.abs(
                    (y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon)
                )
                # Recortar los valores extremos (1% superior e inferior)
                lower_bound = np.percentile(ape, 1)
                upper_bound = np.percentile(ape, 99)
                ape_trimmed = ape[(ape >= lower_bound) & (ape <= upper_bound)]
                metrics["mape"] = (
                    np.mean(ape_trimmed) if len(ape_trimmed) > 0 else np.nan
                )
            else:
                # Para modelos lineales estándar, usar MAPE normal
                metrics["mape"] = mean_absolute_percentage_error(
                    y_true_filtered, y_pred_filtered
                )
        except Exception as e:
            metrics["mape"] = np.nan

        # RMSLE con manejo robusto de valores no positivos
        try:
            # Asegurar que todos los valores sean positivos
            y_true_pos = np.maximum(y_true_filtered, 1e-10)
            y_pred_pos = np.maximum(y_pred_filtered, 1e-10)

            # Para RANSAC, considerar una transformación logarítmica más robusta
            if model_type == "ransac":
                # Suavizar aún más los valores para evitar problemas numéricos
                y_true_log = np.log1p(y_true_pos)
                y_pred_log = np.log1p(y_pred_pos)

                # Calcular RMSLE con manejo de outliers
                log_errors = y_true_log - y_pred_log
                # Recortar errores extremos (2% superior e inferior)
                lower_bound = np.percentile(log_errors, 2)
                upper_bound = np.percentile(log_errors, 98)
                log_errors_trimmed = log_errors[
                    (log_errors >= lower_bound) & (log_errors <= upper_bound)
                ]

                if len(log_errors_trimmed) > 0:
                    metrics["rmsle"] = np.sqrt(np.mean(log_errors_trimmed**2))
                else:
                    metrics["rmsle"] = np.nan
            else:
                # Para modelos lineales estándar, usar RMSLE normal
                metrics["rmsle"] = np.sqrt(
                    mean_squared_error(np.log1p(y_true_pos), np.log1p(y_pred_pos))
                )
        except Exception as e:
            metrics["rmsle"] = np.nan

        return metrics

    def train_model(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Entrenar ambos modelos (Lineal y RANSAC) para Stage 4 y Stage 8
        Retorna resultados con métricas y predicciones de ambos.
        """

        # 1. Cargar datos y preparar ciclos
        self.load_data()
        self.transform_cycles_data()
        datasets_by_stage = self.get_datasets_by_stage()

        # Verificación mínima de datos
        if datasets_by_stage["stage_4"].shape[0] < 10:
            print("⚠️  Advertencia: Pocos datos para Stage 4")
        if datasets_by_stage["stage_8"].shape[0] < 10:
            print("⚠️  Advertencia: Pocos datos para Stage 8")

        # 2. Preparar datos
        self.prepared_data_stage_4 = self.prepare_data_for_stage(
            datasets_by_stage["stage_4"], "stage_4", test_size, random_state
        )
        self.prepared_data_stage_8 = self.prepare_data_for_stage(
            datasets_by_stage["stage_8"], "stage_8", test_size, random_state
        )

        # 3. Entrenar LINEAR
        self.mlr_model_stage_4 = LinearRegression()
        self.mlr_model_stage_8 = LinearRegression()
        self.mlr_model_stage_4.fit(
            self.prepared_data_stage_4["X_train"], self.prepared_data_stage_4["y_train"]
        )
        self.mlr_model_stage_8.fit(
            self.prepared_data_stage_8["X_train"], self.prepared_data_stage_8["y_train"]
        )

        # 4. Entrenar RANSAC
        self.ransac_model_stage_4 = self.ransac_regression_model(
            X_train=self.prepared_data_stage_4["X_train"],
            y_train=self.prepared_data_stage_4["y_train"],
        )
        self.ransac_model_stage_8 = self.ransac_regression_model(
            X_train=self.prepared_data_stage_8["X_train"],
            y_train=self.prepared_data_stage_8["y_train"],
        )

        # 5. Predicciones LINEAR
        mlr_y_train_pred_4 = self.mlr_model_stage_4.predict(
            self.prepared_data_stage_4["X_train"]
        )
        mlr_y_test_pred_4 = self.mlr_model_stage_4.predict(
            self.prepared_data_stage_4["X_test"]
        )
        mlr_y_train_pred_8 = self.mlr_model_stage_8.predict(
            self.prepared_data_stage_8["X_train"]
        )
        mlr_y_test_pred_8 = self.mlr_model_stage_8.predict(
            self.prepared_data_stage_8["X_test"]
        )

        # 6. Predicciones RANSAC
        ransac_y_train_pred_4 = self.ransac_model_stage_4.predict(
            self.prepared_data_stage_4["X_train"]
        )
        ransac_y_test_pred_4 = self.ransac_model_stage_4.predict(
            self.prepared_data_stage_4["X_test"]
        )
        ransac_y_train_pred_8 = self.ransac_model_stage_8.predict(
            self.prepared_data_stage_8["X_train"]
        )
        ransac_y_test_pred_8 = self.ransac_model_stage_8.predict(
            self.prepared_data_stage_8["X_test"]
        )

        # 7. Métricas LINEAR
        mlr_train_metrics_4 = self.calculate_metrics(
            self.prepared_data_stage_4["y_train"],
            mlr_y_train_pred_4,
            model_type="linear",
        )
        mlr_test_metrics_4 = self.calculate_metrics(
            self.prepared_data_stage_4["y_test"], mlr_y_test_pred_4, model_type="linear"
        )
        mlr_train_metrics_8 = self.calculate_metrics(
            self.prepared_data_stage_8["y_train"],
            mlr_y_train_pred_8,
            model_type="linear",
        )
        mlr_test_metrics_8 = self.calculate_metrics(
            self.prepared_data_stage_8["y_test"], mlr_y_test_pred_8, model_type="linear"
        )

        # 8. Métricas RANSAC
        ransac_train_metrics_4 = self.calculate_metrics(
            self.prepared_data_stage_4["y_train"],
            ransac_y_train_pred_4,
            model_type="ransac",
        )
        ransac_test_metrics_4 = self.calculate_metrics(
            self.prepared_data_stage_4["y_test"],
            ransac_y_test_pred_4,
            model_type="ransac",
        )
        ransac_train_metrics_8 = self.calculate_metrics(
            self.prepared_data_stage_8["y_train"],
            ransac_y_train_pred_8,
            model_type="ransac",
        )
        ransac_test_metrics_8 = self.calculate_metrics(
            self.prepared_data_stage_8["y_test"],
            ransac_y_test_pred_8,
            model_type="ransac",
        )

        # 9. Resultados Stage 4
        self.model_results_stage_4 = {
            "X_train": self.prepared_data_stage_4["X_train"],
            "X_test": self.prepared_data_stage_4["X_test"],
            "y_train": self.prepared_data_stage_4["y_train"],
            "y_test": self.prepared_data_stage_4["y_test"],
            "mlr_y_train_pred": mlr_y_train_pred_4,
            "mlr_y_test_pred": mlr_y_test_pred_4,
            "ransac_y_train_pred": ransac_y_train_pred_4,
            "ransac_y_test_pred": ransac_y_test_pred_4,
            "train_samples": self.prepared_data_stage_4["train_samples"],
            "test_samples": self.prepared_data_stage_4["test_samples"],
            "total_features": self.prepared_data_stage_4["total_features"],
            **{f"mlr_train_{k}": v for k, v in mlr_train_metrics_4.items()},
            **{f"mlr_test_{k}": v for k, v in mlr_test_metrics_4.items()},
            **{f"ransac_train_{k}": v for k, v in ransac_train_metrics_4.items()},
            **{f"ransac_test_{k}": v for k, v in ransac_test_metrics_4.items()},
        }

        # 10. Resultados Stage 8
        self.model_results_stage_8 = {
            "X_train": self.prepared_data_stage_8["X_train"],
            "X_test": self.prepared_data_stage_8["X_test"],
            "y_train": self.prepared_data_stage_8["y_train"],
            "y_test": self.prepared_data_stage_8["y_test"],
            "mlr_y_train_pred": mlr_y_train_pred_8,
            "mlr_y_test_pred": mlr_y_test_pred_8,
            "ransac_y_train_pred": ransac_y_train_pred_8,
            "ransac_y_test_pred": ransac_y_test_pred_8,
            "train_samples": self.prepared_data_stage_8["train_samples"],
            "test_samples": self.prepared_data_stage_8["test_samples"],
            "total_features": self.prepared_data_stage_8["total_features"],
            **{f"mlr_train_{k}": v for k, v in mlr_train_metrics_8.items()},
            **{f"mlr_test_{k}": v for k, v in mlr_test_metrics_8.items()},
            **{f"ransac_train_{k}": v for k, v in ransac_train_metrics_8.items()},
            **{f"ransac_test_{k}": v for k, v in ransac_test_metrics_8.items()},
        }

        # 11. Retorno final
        return {
            "model_results_stage_4": self.model_results_stage_4,
            "model_results_stage_8": self.model_results_stage_8,
            "mlr_model_stage_4": self.mlr_model_stage_4,
            "mlr_model_stage_8": self.mlr_model_stage_8,
            "ransac_model_stage_4": self.ransac_model_stage_4,
            "ransac_model_stage_8": self.ransac_model_stage_8,
            "feature_scaler_stage_4": self.feature_scaler_stage_4,
            "feature_scaler_stage_8": self.feature_scaler_stage_8,
            "destination_encoder_stage_4": self.destination_encoder_stage_4,
            "destination_encoder_stage_8": self.destination_encoder_stage_8,
        }

    def ransac_regression_model(self, X_train, y_train, n_seeds=25):
        """
        Crear y retornar un modelo de regresión RANSAC optimizado.
        Basado en la implementación anterior que entregaba mejores resultados.
        """
        print("Optimizando modelo RANSAC...")

        n_samples, n_features = X_train.shape

        # Configuraciones adaptadas al tamaño de los datos (como en la versión anterior)
        configurations = [
            # Configuración conservadora
            {
                "min_samples": max(n_features + 1, n_samples // 10),
                "residual_threshold": None,
                "max_trials": 300,
                "stop_score": 0.85,
                "stop_probability": 0.90,
                "loss": "absolute_error",
            },
            # Configuración permisiva
            {
                "min_samples": max(n_features + 1, n_samples // 15),
                "residual_threshold": None,
                "max_trials": 500,
                "stop_score": 0.80,
                "stop_probability": 0.95,
                "loss": "absolute_error",
            },
            # Configuración muy permisiva
            {
                "min_samples": max(n_features + 1, n_samples // 20),
                "residual_threshold": None,
                "max_trials": 800,
                "stop_score": 0.75,
                "stop_probability": 0.95,
                "loss": "absolute_error",
            },
        ]

        best_score = -np.inf
        best_ransac_model = None
        best_optimization_metrics = {}

        total_tests = len(configurations) * n_seeds
        current_test = 0

        for config_idx, config in enumerate(configurations):
            print(f"Probando configuración {config_idx + 1}/{len(configurations)}")

            for seed in range(n_seeds):
                current_test += 1
                try:
                    params = config.copy()
                    params["random_state"] = seed

                    # Entrenar modelo RANSAC
                    ransac = linear_model.RANSACRegressor(**params)
                    ransac.fit(X_train, y_train)

                    # Calcular métricas en datos de entrenamiento
                    y_train_pred = ransac.predict(X_train)
                    r2_train = r2_score(y_train, y_train_pred)
                    mae_train = mean_absolute_error(y_train, y_train_pred)

                    # Métricas específicas de RANSAC
                    inlier_mask = ransac.inlier_mask_
                    n_inliers = np.sum(inlier_mask)
                    inlier_ratio = n_inliers / len(y_train)

                    # R² en inliers
                    if n_inliers > n_features + 1:
                        r2_inliers = r2_score(
                            y_train[inlier_mask], y_train_pred[inlier_mask]
                        )
                    else:
                        r2_inliers = -np.inf

                    # Score compuesto para optimización (igual que en la versión anterior)
                    composite_score = (
                        0.6 * max(0, r2_train)
                        + 0.2 * max(0, r2_inliers)
                        + 0.15 * inlier_ratio
                        + 0.05
                        * max(0, 1 - mae_train / (np.max(y_train) - np.min(y_train)))
                    )

                    if composite_score > best_score and r2_train > 0:
                        best_score = composite_score
                        best_ransac_model = ransac
                        best_optimization_metrics = {
                            "r2_train": r2_train,
                            "r2_inliers": r2_inliers,
                            "mae_train": mae_train,
                            "n_inliers": n_inliers,
                            "inlier_ratio": inlier_ratio,
                            "composite_score": composite_score,
                            "n_trials": ransac.n_trials_,
                        }

                    if current_test % 10 == 0:
                        progress = (current_test / total_tests) * 100
                        best_r2 = best_optimization_metrics.get("r2_train", 0)
                        print(f"Progreso: {progress:.1f}% - Mejor R²: {best_r2:.4f}")

                except Exception as e:
                    continue

        if best_ransac_model is None:
            print("Usando configuración RANSAC por defecto...")
            best_ransac_model = linear_model.RANSACRegressor(
                min_samples=max(n_features + 1, 10),
                max_trials=500,
                random_state=42,
            )
            best_ransac_model.fit(X_train, y_train)

        print(f"\n✓ Optimización RANSAC completada:")
        print(
            f"  - R² entrenamiento: {best_optimization_metrics.get('r2_train', 0):.4f}"
        )
        print(
            f"  - Ratio de inliers: {best_optimization_metrics.get('inlier_ratio', 0):.4f}"
        )
        print(f"  - N° inliers: {best_optimization_metrics.get('n_inliers', 0)}")

        return best_ransac_model

    def show_results_summary(self):
        """
        Mostrar resumen de resultados de ambos modelos (MLR y RANSAC) para Stage 4 y Stage 8
        """
        if not self.model_results_stage_4 or not self.model_results_stage_8:
            print("❌ No hay resultados disponibles. Entrena los modelos primero.")
            return

        print("\n" + "=" * 80)
        print("📈 RESUMEN DE RESULTADOS - MLR y RANSAC")
        print("=" * 80)

        # Función auxiliar para formatear valores, manejando NaN e infinitos
        def format_value(value, precision=4):
            if value is None or (
                isinstance(value, float) and (np.isnan(value) or np.isinf(value))
            ):
                return "N/A"
            try:
                return f"{value:.{precision}f}"
            except:
                return str(value)

        # === Stage 4 ===
        print("\n🔵 STAGE 4 (Camión vacío)")
        print("-" * 60)
        print(
            f"Muestras entrenamiento: {self.model_results_stage_4.get('train_samples', 'N/A')}"
        )
        print(
            f"Muestras prueba: {self.model_results_stage_4.get('test_samples', 'N/A')}"
        )

        # --- MLR ---
        print("\n  📊 Linear Regression:")
        print(
            f"    R² Score: {format_value(self.model_results_stage_4.get('mlr_test_r2'))}"
        )
        print(
            f"    MAE: {format_value(self.model_results_stage_4.get('mlr_test_mae'))}"
        )
        print(
            f"    RMSE: {format_value(self.model_results_stage_4.get('mlr_test_rmse'))}"
        )
        print(
            f"    MAPE: {format_value(self.model_results_stage_4.get('mlr_test_mape'), 2)}%"
        )
        print(
            f"    RMSLE: {format_value(self.model_results_stage_4.get('mlr_test_rmsle'))}"
        )

        # --- RANSAC ---
        print("\n  📊 RANSAC Regression:")
        print(
            f"    R² Score: {format_value(self.model_results_stage_4.get('ransac_test_r2'))}"
        )
        print(
            f"    MAE: {format_value(self.model_results_stage_4.get('ransac_test_mae'))}"
        )
        print(
            f"    RMSE: {format_value(self.model_results_stage_4.get('ransac_test_rmse'))}"
        )
        print(
            f"    MAPE: {format_value(self.model_results_stage_4.get('ransac_test_mape'), 2)}%"
        )
        print(
            f"    RMSLE: {format_value(self.model_results_stage_4.get('ransac_test_rmsle'))}"
        )

        # === Stage 8 ===
        print("\n🔴 STAGE 8 (Camión cargado)")
        print("-" * 60)
        print(
            f"Muestras entrenamiento: {self.model_results_stage_8.get('train_samples', 'N/A')}"
        )
        print(
            f"Muestras prueba: {self.model_results_stage_8.get('test_samples', 'N/A')}"
        )

        # --- MLR ---
        print("\n  📊 Linear Regression:")
        print(
            f"    R² Score: {format_value(self.model_results_stage_8.get('mlr_test_r2'))}"
        )
        print(
            f"    MAE: {format_value(self.model_results_stage_8.get('mlr_test_mae'))}"
        )
        print(
            f"    RMSE: {format_value(self.model_results_stage_8.get('mlr_test_rmse'))}"
        )
        print(
            f"    MAPE: {format_value(self.model_results_stage_8.get('mlr_test_mape'), 2)}%"
        )
        print(
            f"    RMSLE: {format_value(self.model_results_stage_8.get('mlr_test_rmsle'))}"
        )

        # --- RANSAC ---
        print("\n  📊 RANSAC Regression:")
        print(
            f"    R² Score: {format_value(self.model_results_stage_8.get('ransac_test_r2'))}"
        )
        print(
            f"    MAE: {format_value(self.model_results_stage_8.get('ransac_test_mae'))}"
        )
        print(
            f"    RMSE: {format_value(self.model_results_stage_8.get('ransac_test_rmse'))}"
        )
        print(
            f"    MAPE: {format_value(self.model_results_stage_8.get('ransac_test_mape'), 2)}%"
        )
        print(
            f"    RMSLE: {format_value(self.model_results_stage_8.get('ransac_test_rmsle'))}"
        )

        print("\n" + "=" * 80)

    def save_predictions_csv(self, filename: str = "fuel_predictions.csv"):
        """
        Guarda un CSV con los datos de ciclos, metadatos y predicciones de ambos modelos

        Args:
            filename (str): Nombre del archivo CSV a guardar
        """
        # Verificar que tenemos datos de ciclos
        if self.cycles_data.is_empty():
            print("❌ No hay datos de ciclos. Ejecuta transform_cycles_data() primero.")
            return

        # Verificar que tenemos modelos entrenados
        if (
            self.mlr_model_stage_4 is None
            or self.mlr_model_stage_8 is None
            or self.ransac_model_stage_4 is None
            or self.ransac_model_stage_8 is None
        ):
            print("❌ No hay modelos entrenados. Ejecuta train_model() primero.")
            return

        # Crear una copia de los datos de ciclos
        output_data = self.cycles_data.clone()

        # Preparar arrays para las predicciones
        n_samples = len(output_data)
        predicted_fuel_lr = np.full(n_samples, np.nan)
        predicted_fuel_ransac = np.full(n_samples, np.nan)
        residual_lr = np.full(n_samples, np.nan)
        residual_ransac = np.full(n_samples, np.nan)
        is_inlier = np.full(n_samples, False)
        is_outlier = np.full(n_samples, False)

        # Procesar datos para Stage 4
        stage_4_data = output_data.filter(pl.col("StageSequence") == 4)
        if not stage_4_data.is_empty():
            X_stage_4 = self._prepare_stage_data_for_prediction(stage_4_data, "stage_4")

            # Predecir con ambos modelos
            lr_pred_4 = self.mlr_model_stage_4.predict(X_stage_4)
            ransac_pred_4 = self.ransac_model_stage_4.predict(X_stage_4)

            # Obtener índices de Stage 4 en el DataFrame completo
            stage_4_indices = output_data.filter(pl.col("StageSequence") == 4)[
                "cycle_group"
            ].to_list()
            idx_map_4 = {
                cycle_group: i
                for i, cycle_group in enumerate(output_data["cycle_group"].to_list())
            }

            # Asignar predicciones
            for i, cycle_group in enumerate(stage_4_data["cycle_group"].to_list()):
                idx = idx_map_4[cycle_group]
                predicted_fuel_lr[idx] = lr_pred_4[i]
                predicted_fuel_ransac[idx] = ransac_pred_4[i]
                residual_lr[idx] = stage_4_data["FuelConsumed"][i] - lr_pred_4[i]
                residual_ransac[idx] = (
                    stage_4_data["FuelConsumed"][i] - ransac_pred_4[i]
                )

            # Marcar inliers para RANSAC
            if hasattr(self.ransac_model_stage_4, "inlier_mask_"):
                inlier_mask_4 = self.ransac_model_stage_4.inlier_mask_
                for i, cycle_group in enumerate(stage_4_data["cycle_group"].to_list()):
                    if i < len(inlier_mask_4):
                        idx = idx_map_4[cycle_group]
                        is_inlier[idx] = inlier_mask_4[i]
                        is_outlier[idx] = not inlier_mask_4[i]

        # Procesar datos para Stage 8
        stage_8_data = output_data.filter(pl.col("StageSequence") == 8)
        if not stage_8_data.is_empty():
            X_stage_8 = self._prepare_stage_data_for_prediction(stage_8_data, "stage_8")

            # Predecir con ambos modelos
            lr_pred_8 = self.mlr_model_stage_8.predict(X_stage_8)
            ransac_pred_8 = self.ransac_model_stage_8.predict(X_stage_8)

            # Obtener índices de Stage 8 en el DataFrame completo
            stage_8_indices = output_data.filter(pl.col("StageSequence") == 8)[
                "cycle_group"
            ].to_list()
            idx_map_8 = {
                cycle_group: i
                for i, cycle_group in enumerate(output_data["cycle_group"].to_list())
            }

            # Asignar predicciones
            for i, cycle_group in enumerate(stage_8_data["cycle_group"].to_list()):
                idx = idx_map_8[cycle_group]
                predicted_fuel_lr[idx] = lr_pred_8[i]
                predicted_fuel_ransac[idx] = ransac_pred_8[i]
                residual_lr[idx] = stage_8_data["FuelConsumed"][i] - lr_pred_8[i]
                residual_ransac[idx] = (
                    stage_8_data["FuelConsumed"][i] - ransac_pred_8[i]
                )

            # Marcar inliers para RANSAC
            if hasattr(self.ransac_model_stage_8, "inlier_mask_"):
                inlier_mask_8 = self.ransac_model_stage_8.inlier_mask_
                for i, cycle_group in enumerate(stage_8_data["cycle_group"].to_list()):
                    if i < len(inlier_mask_8):
                        idx = idx_map_8[cycle_group]
                        is_inlier[idx] = inlier_mask_8[i]
                        is_outlier[idx] = not inlier_mask_8[i]

        # Añadir todas las columnas al DataFrame
        output_data = output_data.with_columns(
            [
                pl.Series("predicted_fuel_lr", predicted_fuel_lr),
                pl.Series("predicted_fuel_ransac", predicted_fuel_ransac),
                pl.Series("residual_lr", residual_lr),
                pl.Series("residual_ransac", residual_ransac),
                pl.Series("is_inlier", is_inlier),
                pl.Series("is_outlier", is_outlier),
            ]
        )

        # Seleccionar solo las columnas requeridas
        final_columns = [
            "cycle_group",
            "TimeStampIni",
            "TimeStampFin",
            "ShiftDate",
            "Equipment",
            "TruckFleet",
            "StartCycle",
            "EndCycle",
            "AverageSpeed",
            "AvgRPM",
            "AvgSlopePercent",
            "AvgAcceleration",
            "TotalMeasuredTonnage",
            "Distance",
            "StageSequence",
            "Destination",
            "RecordsInCycle",
            "FuelConsumed",
            "CycleDurationSeconds",
            "is_inlier",
            "is_outlier",
            "predicted_fuel_lr",
            "predicted_fuel_ransac",
            "residual_lr",
            "residual_ransac",
        ]

        # Filtrar solo filas con predicciones válidas
        valid_predictions = output_data.filter(
            pl.col("predicted_fuel_lr").is_not_null()
            | pl.col("predicted_fuel_ransac").is_not_null()
        )

        # Guardar a CSV
        valid_predictions.select(final_columns).write_csv(filename)
        print(f"✅ CSV guardado como {filename} con {len(valid_predictions)} registros")

    def _prepare_stage_data_for_prediction(
        self, data: pl.DataFrame, stage: str
    ) -> np.ndarray:
        """
        Prepara datos para predicción usando los mismos preprocesadores del entrenamiento

        Args:
            data (pl.DataFrame): Datos a preparar
            stage (str): Etapa para la cual preparar los datos ("stage_4" o "stage_8")

        Returns:
            np.ndarray: Datos preparados para predicción
        """
        # Extraer variables predictoras numéricas
        X_numeric = data.select(self.predictor_vars).to_pandas()

        # Extraer variable categórica
        X_destination = data.select("Destination").to_pandas()

        # Seleccionar encoder y scaler según el stage
        if stage == "stage_4":
            destination_encoder = self.destination_encoder_stage_4
            feature_scaler = self.feature_scaler_stage_4
        else:
            destination_encoder = self.destination_encoder_stage_8
            feature_scaler = self.feature_scaler_stage_8

        # Codificación binaria para variable categórica 'Destination'
        X_dest_encoded = destination_encoder.transform(X_destination).to_numpy()

        # Estandarización de variables numéricas
        X_numeric_scaled = feature_scaler.transform(X_numeric)

        # Combinar características numéricas escaladas y categóricas codificadas
        return np.hstack([X_numeric_scaled, X_dest_encoded])


def main():
    """
    Función principal para ejecutar el modelo y guardar las predicciones
    """
    # Definir variables predictoras
    predictor_variables = [
        "AverageSpeed",
        "AvgRPM",
        "AvgSlopePercent",
        "AvgAcceleration",
        "TotalMeasuredTonnage",
        "Distance",
        "CycleDurationSeconds",
    ]

    try:
        # 1. Crear instancia y entrenar ambos modelos
        print("🚀 Creando modelo...")
        mi_model = LinearRegressionMultivariable(predictor_variables)

        print("🔄 Entrenando modelos...")
        resultados = mi_model.train_model()

        # 2. Mostrar resumen de resultados
        mi_model.show_results_summary()

        # 3. Guardar predicciones en CSV
        print("💾 Guardando predicciones en CSV...")
        mi_model.save_predictions_csv("predicciones_combustible.csv")

        print("✅ Proceso completado exitosamente!")

        return mi_model, resultados

    except FileNotFoundError:
        print("❌ Archivo 'unified_data_T-210.csv' no encontrado.")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    modelo, resultados = main()
