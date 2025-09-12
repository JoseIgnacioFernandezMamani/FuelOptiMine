import polars as pl
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("lrm.log", mode="a"),  # guarda en archivo
        logging.StreamHandler(),  # también imprime en consola
    ],
)
logger = logging.getLogger(__name__)


class LinearRegressionModel:
    """
    Clase simplificada para regresión lineal con dos modelos: Stage 4 y Stage 8
    """

    def __init__(self):
        self.cycles_data: pl.DataFrame = pl.DataFrame()
        self.predictor_vars: list[str] = [
            "Distance",
            "CycleDurationSeconds",  # no incluyo tonage porque no aporta nada realmente a la prediccion.
        ]

        # Escaladores para cada stage
        self.scaler_stage_4: StandardScaler = StandardScaler()
        self.scaler_stage_8: StandardScaler = StandardScaler()

        # Modelos
        self.model_stage_4: linear_model.RANSACRegressor = (
            linear_model.RANSACRegressor()
        )
        self.model_stage_8: linear_model.RANSACRegressor = (
            linear_model.RANSACRegressor()
        )

        # modelo de isolation forest para detección de outliers
        self.iso_forest_stage_4: IsolationForest = IsolationForest(
            contamination=0.05, random_state=42
        )
        self.iso_forest_stage_8: IsolationForest = IsolationForest(
            contamination=0.05, random_state=42
        )

        # Resultados
        self.results_stage_4: dict = {}
        self.results_stage_8: dict = {}
        self.df: pl.DataFrame = pl.DataFrame()

    def load_data(self):
        """
        Load data from CSV file (hardcoding for now).
        """
        logger.info("Cargando datos desde ...")
        df = pl.read_csv("unified_data_T-210.csv", try_parse_dates=True)
        self.df = df.sort("SortTimestamp")

    def transform_cycles_data(self):
        """
        Process data to identify cycles and calculate fuel consumption metrics

        This method performs comprehensive data transformation to identify truck operational
        cycles and calculate fuel consumption for each cycle. It processes sensor data to
        create meaningful cycle-based aggregations for model training.

        The transformation includes:
            - Cycle identification based on stage sequences
            - Fuel level smoothing using rolling median
            - Geographic data unification
            - Cycle grouping and aggregation
            - Fuel consumption calculation
            - Data quality filtering

        Results are stored in self.cycles_data as a Polars DataFrame.
        """
        df = self.df.with_columns(
            [
                # identificar ciclos
                pl.when((pl.col("StageSequence") == 4) | (pl.col("StageSequence") == 8))
                .then(True)
                .otherwise(False)
                .alias("cycle_end"),
                # rolling median del nivel de combustible
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=10, min_samples=3, center=True)
                .alias("MedianFuelLevelLiters"),
                # unificar destinos
                pl.coalesce([pl.col("LoadingZone"), pl.col("Destination")]).alias(
                    "Destination"
                ),
                # datos geográficos
                pl.when(pl.col("Latitude") != 0)
                .then(pl.col("Latitude"))
                .otherwise(pl.col("Latitude_cycle"))
                .alias("Latitude"),
                pl.when(pl.col("Longitude") != 0)
                .then(pl.col("Longitude"))
                .otherwise(pl.col("Longitude_cycle"))
                .alias("Longitude"),
                pl.when(pl.col("Elevation") != 0)
                .then(pl.col("Elevation"))
                .otherwise(pl.col("Elevation_cycle"))
                .alias("Elevation"),
            ]
        )

        # crear grupos de ciclos
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
                    pl.col("TimeStamp").first().alias("TimeStampIni"),
                    pl.col("TimeStamp").last().alias("TimeStampFin"),
                    pl.col("ShiftDate").last().alias("ShiftDate"),
                    pl.col("Shift").last().alias("Shift"),
                    pl.col("Equipment").last().alias("Equipment"),
                    pl.col("TruckFleet")
                    .last()
                    .alias("TruckFleet"),  # fin de los metadatos
                    # variables para calculo
                    pl.col("MedianFuelLevelLiters")
                    .first()
                    .alias("StartCycle"),  # para calculo inicial
                    pl.col("MedianFuelLevelLiters")
                    .last()
                    .alias("EndCycle"),  # para calculo final
                    # variables numericas para modelo
                    pl.col("SpeedAvg").mean().alias("AvgSpeed"),
                    pl.col("SlopePercent").mean().alias("AvgSlopePercent"),
                    pl.col("Acceleration").mean().alias("AvgAcceleration"),
                    pl.col("TimeEfficiencyPercentage")
                    .mean()
                    .alias("AvgTimeEfficiencyPercentage"),
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
            .sort("TimeStampIni")
        )

        # calcular combustible consumido y duración del ciclo
        result = result.with_columns(
            [
                pl.when(
                    ((pl.col("StartCycle") - pl.col("EndCycle")).abs() <= 500)
                    & ((pl.col("StartCycle") - pl.col("EndCycle")).abs() >= 5)
                )
                .then((pl.col("StartCycle") - pl.col("EndCycle")).abs())
                .when((pl.col("StartCycle") - pl.col("EndCycle")) < 5)
                .then(5)
                .alias("FuelConsumed"),
                (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                .dt.total_seconds()
                .abs()
                .alias("CycleDurationSeconds"),
            ]
        )

        # filtros básicos
        result = result.filter(
            (pl.col("Destination").str.strip_chars().str.len_bytes() > 2)
            & (pl.col("Distance") > 0)
            & (pl.col("TotalMeasuredTonnage") >= 0)
            & (pl.col("CycleDurationSeconds") > 120)
            & (pl.col("CycleDurationSeconds") < 21600)
            & (pl.col("FuelConsumed") >= 0.1)
            & (pl.col("FuelConsumed") <= 210)
            & (pl.col("StageSequence").is_not_null())
        )

        self.cycles_data = result

    def prepare_data_for_stage(
        self,
        data: pl.DataFrame,
        scaler: StandardScaler,
        iso_forest: IsolationForest,
        test_size: float = 0.2,
        random_state: int = 42,
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
        logger.info(f"Preparando datos para stage con {len(data)} muestras iniciales")

        # select predictor variables and target
        X = data.select(self.predictor_vars).to_pandas()
        y = data.select("FuelConsumed").to_numpy().flatten()

        logger.info(f"Variables predictoras: {self.predictor_vars}")

        # delete outliers using IsolationForest
        logger.info("Detectando outliers con Isolation Forest...")
        outlier_predictions = iso_forest.fit_predict(X)

        # Create boolean mask for inliers (1 = inlier, -1 = outlier)
        inlier_mask = outlier_predictions == 1
        n_outliers = np.sum(~inlier_mask)
        outlier_percentage = (n_outliers / len(X)) * 100

        logger.info(f"Outliers detectados: {n_outliers} ({outlier_percentage:.2f}%)")

        # filter to keep only inliers
        X_clean = X[inlier_mask]
        y_clean = y[inlier_mask]

        logger.info(
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

        logger.info(
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

    def ransac_regression_model(self, X_train, y_train, n_seeds=25):
        """
        Create and return an optimized RANSAC regression model

        This method performs hyperparameter optimization for RANSAC regression by testing
        multiple configurations with different random seeds to find the best performing model.

        Args:
            X_train: Training feature matrix
            y_train: Training target values
            n_seeds (int, optional): Number of random seeds to test per configuration.
                                    Defaults to 25.

        Returns:
            RANSACRegressor: Best performing RANSAC model fitted on training datCrear y retornar un modelo de regresión RANSAC optimizado.
        """
        logger.info("Iniciando optimización de modelo RANSAC...")
        n_samples, n_features = X_train.shape

        # Define multiple RANSAC configurations with different hyperparameters
        configurations = [
            {
                "min_samples": max(n_features + 1, n_samples // 10),
                "residual_threshold": None,
                "max_trials": 300,
                "stop_score": 0.85,
                "stop_probability": 0.90,
                "loss": "absolute_error",
            },
            {
                "min_samples": max(n_features + 1, n_samples // 15),
                "residual_threshold": None,
                "max_trials": 500,
                "stop_score": 0.80,
                "stop_probability": 0.95,
                "loss": "absolute_error",
            },
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
        last_logged_metrics = {}

        total_tests = len(configurations) * n_seeds
        current_test = 0

        for config_idx, config in enumerate(configurations):
            logger.info(
                f"Probando configuración {config_idx + 1}/{len(configurations)}"
            )

            for seed in range(n_seeds):
                current_test += 1
                try:
                    # Create RANSAC model with current configuration and seed
                    params = config.copy()
                    params["random_state"] = seed
                    ransac = linear_model.RANSACRegressor(**params)
                    ransac.fit(X_train, y_train)

                    # Calculate training performance metrics
                    y_train_pred = ransac.predict(X_train)
                    r2_train = r2_score(y_train, y_train_pred)
                    mae_train = mean_absolute_error(y_train, y_train_pred)

                    # Calculate RANSAC-specific effectiveness metrics
                    inlier_mask = ransac.inlier_mask_  # Boolean mask of inliers
                    n_inliers = np.sum(inlier_mask)  # Count of inlier points
                    inlier_ratio = n_inliers / len(y_train)  # Proportion of inliers

                    # Calculate R² on inliers only (model's core performance)
                    r2_inliers = (
                        r2_score(y_train[inlier_mask], y_train_pred[inlier_mask])
                        if n_inliers > n_features + 1
                        else -np.inf
                    )

                    # Composite scoring function combining multiple criteria
                    composite_score = (
                        0.6 * max(0, r2_train)  # Overall fit quality (60%)
                        + 0.2 * max(0, r2_inliers)  # Inlier fit quality (20%)
                        + 0.15 * inlier_ratio  # Inlier proportion (15%)
                        + 0.05
                        * max(
                            0, 1 - mae_train / (np.max(y_train) - np.min(y_train))
                        )  # Normalized MAE (5%)
                    )

                    # Update best model if current one is better and has positive R²
                    if composite_score > best_score and r2_train > 0:
                        best_score = composite_score
                        best_ransac_model = ransac
                        current_metrics = {
                            "r2_train": r2_train,
                            "r2_inliers": r2_inliers,
                            "mae_train": mae_train,
                            "n_inliers": n_inliers,
                            "inlier_ratio": inlier_ratio,
                            "composite_score": composite_score,
                            "n_trials": ransac.n_trials_,  # Actual trials used by RANSAC
                        }

                        # Determine if this represents a significant improvement
                        has_improvement = False
                        if not last_logged_metrics:
                            has_improvement = True  # first valid found
                        else:
                            # Check for improvement in key metrics
                            if (
                                current_metrics["r2_train"]
                                > last_logged_metrics.get("r2_train", -np.inf)
                                or current_metrics["composite_score"]
                                > last_logged_metrics.get("composite_score", -np.inf)
                                or current_metrics["mae_train"]
                                < last_logged_metrics.get("mae_train", np.inf)
                            ):
                                has_improvement = True
                        # Log improvement details
                        if has_improvement:
                            best_optimization_metrics = current_metrics
                            last_logged_metrics = current_metrics.copy()

                            logger.info("==== MEJORA ENCONTRADA ====")
                            logger.info(
                                f"Test {current_test}/{total_tests} - Nueva mejor configuración:"
                            )
                            logger.info(f"R² Train: {current_metrics['r2_train']:.4f}")
                            logger.info(
                                f"R² Inliers: {current_metrics['r2_inliers']:.4f}"
                            )
                            logger.info(
                                f"MAE Train: {current_metrics['mae_train']:.4f}"
                            )
                            logger.info(
                                f"Inliers: {current_metrics['n_inliers']} ({current_metrics['inlier_ratio']:.4f})"
                            )
                            logger.info(
                                f"Score Compuesto: {current_metrics['composite_score']:.4f}"
                            )
                            logger.info(f"Trials: {current_metrics['n_trials']}")
                except Exception as e:
                    # Log errors for debugging but continue optimization
                    logger.debug(
                        f"Error en configuración {config_idx}, seed {seed}: {str(e)}"
                    )
                    continue

        # Fallback to default RANSAC if optimization failed
        if best_ransac_model is None:
            logger.info("Usando configuración RANSAC por defecto...")
            best_ransac_model = linear_model.RANSACRegressor(
                min_samples=max(n_features + 1, 10),
                max_trials=500,
                random_state=42,
            )
            best_ransac_model.fit(X_train, y_train)

        return best_ransac_model

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
        stage_4_data = self.cycles_data.filter(pl.col("StageSequence") == 4)
        stage_8_data = self.cycles_data.filter(pl.col("StageSequence") == 8)

        logger.info("Entrenando el modelo stage_4")

        # Prepare and scale Stage 4 data for training
        stage_4_scaled_data = self.prepare_data_for_stage(
            stage_4_data,
            self.scaler_stage_4,
            self.iso_forest_stage_4,
            test_size,
            random_state,
        )
        self.model_stage_4 = self.ransac_regression_model(
            stage_4_scaled_data["X_train"], stage_4_scaled_data["y_train"]
        )

        y_pred_train_4 = self.model_stage_4.predict(stage_4_scaled_data["X_train"])
        y_pred_test_4 = self.model_stage_4.predict(stage_4_scaled_data["X_test"])

        logger.info("Entrenando el modelo stage_8")

        # Prepare and scale Stage 8 data for training
        stage_8_scaled_data = self.prepare_data_for_stage(
            stage_8_data,
            self.scaler_stage_8,
            self.iso_forest_stage_8,
            test_size,
            random_state,
        )
        self.model_stage_8 = self.ransac_regression_model(
            stage_8_scaled_data["X_train"], stage_8_scaled_data["y_train"]
        )
        y_pred_train_8 = self.model_stage_8.predict(stage_8_scaled_data["X_train"])
        y_pred_test_8 = self.model_stage_8.predict(stage_8_scaled_data["X_test"])

        logger.info("Entrenamiento completado.")

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
                "model_parameters": self.inverse_scale(
                    self.model_stage_4, self.scaler_stage_4
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
                "model_parameters": self.inverse_scale(
                    self.model_stage_8, self.scaler_stage_8
                ),
                "outlier_info": stage_8_scaled_data["outlier_info"],
            },
            "multicollinearity": self.analyze_multicollinearity(),
        }

        self.log_results(stage="all", results=result)

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

    def analyze_multicollinearity(self) -> dict:
        """
        Analyze multicollinearity among predictor variables using:
            - Pearson correlation matrix
            - Variance Inflation Factor (VIF)
        Applied directly on self.cycles_data (unscaled data).
        """
        df = self.cycles_data[self.predictor_vars].drop_nulls().drop_nans().to_pandas()

        # Correlation matrix
        corr_matrix = df.corr(method="pearson").to_dict()

        # VIF
        vif_data = pd.DataFrame()
        vif_data["Variable"] = df.columns
        vif_data = {
            df.columns[i]: variance_inflation_factor(df.values, i)
            for i in range(df.shape[1])
        }

        return {"correlation_matrix": corr_matrix, "vif": vif_data}

    def log_results(self, stage: str, results: dict):
        """
        Logs all training results in a structured way.
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "stage": stage,
            "predictors": self.predictor_vars,
            "results": results,
        }

        # save as json
        logger.info(
            "Training summary:\n%s", json.dumps(log_entry, indent=4, default=str)
        )

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
        if self.scaler_stage_4 is None or self.scaler_stage_8 is None:
            raise ValueError("Scalers not available. Run train_models() first.")

        logger.info("Generating predictions for Stage 4 and Stage 8...")

        predictions = []

        # Stage 4
        stage_4_data = self.cycles_data.filter(pl.col("StageSequence") == 4)
        if len(stage_4_data) > 0:
            features = stage_4_data.select(self.predictor_vars).to_pandas()
            scaled = self.scaler_stage_4.transform(features)
            preds = self.model_stage_4.predict(scaled)

            stage_4_pred = stage_4_data.with_columns(
                pl.Series("PredictedFuelConsumption", preds)
            )
            predictions.append(stage_4_pred)

            logger.info(f"Generated {len(preds)} predictions for Stage 4")

        # Stage 8
        stage_8_data = self.cycles_data.filter(pl.col("StageSequence") == 8)
        if len(stage_8_data) > 0:
            features = stage_8_data.select(self.predictor_vars).to_pandas()
            scaled = self.scaler_stage_8.transform(features)
            preds = self.model_stage_8.predict(scaled)

            stage_8_pred = stage_8_data.with_columns(
                pl.Series("PredictedFuelConsumption", preds)
            )
            predictions.append(stage_8_pred)

            logger.info(f"Generated {len(preds)} predictions for Stage 8")

        if not predictions:
            raise RuntimeError("No Stage 4 or Stage 8 data available for predictions.")

        # Concatenar y ordenar por TimeStampIni
        predictions_df = pl.concat(predictions).sort("TimeStampIni")

        logger.info(f"Final predictions dataframe shape: {predictions_df.shape}")
        return predictions_df

    def get_cycle_data(self) -> pl.DataFrame:
        """
        Returns the processed cycles data used for training and predictions.
        """
        if self.cycles_data is None:
            raise ValueError("Cycle data not available. Run train_models() first.")
        return self.cycles_data


if __name__ == "__main__":
    model = LinearRegressionModel()
    results = model.train_models()
    predictions_df = model.get_predictions()
    predictions_df.write_csv("predicted_cycles.csv")
    logger.info("Predictions saved to predicted_cycles.csv")
