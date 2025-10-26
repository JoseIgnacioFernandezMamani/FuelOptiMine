import polars as pl
import numpy as np
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    mean_squared_log_error,
)
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Any
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from .linear_regression_multivariable import LinearRegressionModel
from model.utils.model_utils import analyze_multicollinearity, log_results, get_logger


class XGBoostModel:
    """
    Class for training TWO specialized XGBoost models:
    - One for Stage 4 (empty truck)
    - One for Stage 8 (loaded truck)
    Each model uses only its relevant features for better performance.
    """

    def __init__(
        self,
        truck_id: str,
        numeric_predictor_vars: List[str],
        categorical_vars: List[str],
        max_cat_to_onehot: int = 4,
    ):
        """
        Initialize dual models with native categorical support.
        """
        self.categorical_vars = categorical_vars
        self.numeric_predictor_vars = numeric_predictor_vars
        self.df = pl.DataFrame()
        self.cycles_data = pl.DataFrame()

        # MODEL FOR STAGE 4 (Empty Truck) - Simpler behavior
        self.model_stage4 = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.1,
            max_depth=4,  # Less depth for simpler patterns
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=42,
            eval_metric="mae",
            early_stopping_rounds=100,
            reg_alpha=0.01,
            reg_lambda=1,
            verbosity=1,
            enable_categorical=True,
            max_cat_to_onehot=max_cat_to_onehot,
        )

        # MODEL FOR STAGE 8 (Loaded Truck) - More complex behavior
        self.model_stage8 = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=7,  # More depth for complex patterns
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda",
            random_state=42,
            eval_metric="mae",
            early_stopping_rounds=100,
            reg_alpha=0.01,
            reg_lambda=1,
            verbosity=1,
            enable_categorical=True,
            max_cat_to_onehot=max_cat_to_onehot,
        )

        # Legacy compatibility
        self.model = None

        # Feature lists for each stage
        self.features_stage4 = []
        self.features_stage8 = []

        # Results and testing data
        self.test_data = {}
        self.feature_names = []

        # Outlier detection models (one per stage)
        self.iso_forest_stage_4 = IsolationForest()
        self.iso_forest_stage_8 = IsolationForest()

        # Logger
        self.logger = get_logger("XGBoost_Dual", "xgb_dual.log", console=True)

        # RMSE thresholds for decision making
        self.rmse_threshold_st4 = 0
        self.rmse_threshold_st8 = 0

        # Total fuel consumed
        self.total_consumed = 0.0
        self.st1_fuel_consumed: float = 0.0

        # equipment
        self.truck_id = truck_id

    def load_data(self):
        """Load data from LinearRegressionModel"""
        model = LinearRegressionModel(truck_id=self.truck_id)
        results = model.train_models()
        self.total_consumed = results["total_consumed_fuel"]
        self.rmse_threshold_st4 = results["stage_4"]["train_metrics"]["mae"]
        self.rmse_threshold_st8 = results["stage_8"]["train_metrics"]["mae"]
        self.df = model.get_predictions()

    def transform_cycles_data(self):
        """Process data to identify cycles and calculate fuel consumption metrics."""
        self.logger.info("Transformando datos de cycle")

        df = self.df

        result = df.with_columns(
            [
                pl.when(
                    (pl.col("PredictedFuelConsumption") > 100)
                    & (pl.col("FuelConsumed") < 100)
                )
                .then((pl.col("FuelConsumed") + pl.col("PredictedFuelConsumption")) / 2)
                .when(
                    (pl.col("PredictedFuelConsumption") > 100)
                    & (pl.col("FuelConsumed") > 100)
                )
                .then(pl.col("PredictedFuelConsumption") / 2)
                .otherwise(pl.col("PredictedFuelConsumption"))
                .alias("FuelConsumed")
            ]
        )

        cols_to_clean = self.numeric_predictor_vars + ["FuelConsumed"]
        cols_to_clean.remove("StageSequence")

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
        self.logger.info(f"Datos procesados: {len(result)} ciclos válidos.")

    def _prepare_stage_data(self, df_stage, stage_name):
        """
        Helper function to prepare data for a specific stage.
        Returns cleaned data after outlier removal.
        """
        if df_stage.empty:
            self.logger.warning(f"⚠️ No hay datos para {stage_name}")
            return None

        self.logger.info(f"📦 Procesando {stage_name} con {len(df_stage)} registros")

        # Prepare features for Isolation Forest
        X_stage_iso = df_stage[self.numeric_predictor_vars].values
        y_stage = df_stage["FuelConsumed"].values
        X_with_target = np.hstack([X_stage_iso, y_stage.reshape(-1, 1)])

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=200,
            max_samples="auto",
        )
        iso_forest.fit(X_with_target)
        mask = iso_forest.predict(X_with_target) == 1

        n_outliers = np.sum(~mask)
        outlier_pct = (n_outliers / len(df_stage)) * 100
        self.logger.info(
            f"  {stage_name}: Eliminados {n_outliers} outliers ({outlier_pct:.2f}%)"
        )

        # Store the trained isolation forest
        if stage_name == "Stage 4":
            self.iso_forest_stage_4 = iso_forest
        else:
            self.iso_forest_stage_8 = iso_forest

        return df_stage[mask].copy()

    def prepare_data(self):
        """
        Prepare separate datasets for Stage 4 and Stage 8.
        Returns a dictionary with train/test splits for each stage.
        """
        self.logger.info("🔧 Preparando datos para entrenamiento dual...")

        df = self.cycles_data.to_pandas().copy()

        # Split numeric and categorical features
        X_numeric = df[self.numeric_predictor_vars]
        X_categorical = df[self.categorical_vars]
        y = df["FuelConsumed"]

        # Assign categorical dtype
        for cat_col in self.categorical_vars:
            X_categorical[cat_col] = X_categorical[cat_col].astype("category")
            self.logger.info(
                f"  Variable '{cat_col}' convertida a categoría con {X_categorical[cat_col].nunique()} valores únicos"
            )

        # Combine features
        df_full = pd.concat([X_numeric, X_categorical], axis=1)
        df_full["FuelConsumed"] = y
        df_full["StageSequence"] = df["StageSequence"]

        # SEPARATE BY STAGE
        df_stage4 = df_full[df_full["StageSequence"] == 4].copy()
        df_stage8 = df_full[df_full["StageSequence"] == 8].copy()

        self.logger.info(f"\n📊 Distribución de datos:")
        self.logger.info(f"  Stage 4 (vacío): {len(df_stage4)} registros")
        self.logger.info(f"  Stage 8 (lleno): {len(df_stage8)} registros")

        # Clean outliers for each stage
        df_stage4_clean = self._prepare_stage_data(df_stage4, "Stage 4")
        df_stage8_clean = self._prepare_stage_data(df_stage8, "Stage 8")

        # Define specific features for each stage
        # Stage 4: Only features available for empty trucks
        features_s4_numeric = [
            "SpeedAvg",
            "Distance",
            "CycleDurationSeconds",
            "TimeEfficiencyPercentage",
        ]
        features_s4_categorical = [
            "Shovel",
            "Destination",
            "DestinationType",
        ]

        # Stage 8: Only features available for loaded trucks
        features_s8_numeric = [
            "SpeedAvg",
            "TotalMeasuredTonnage",
            "Distance",
            "CycleDurationSeconds",
            "TimeEfficiencyPercentage",
        ]
        features_s8_categorical = ["Destination", "DestinationType", "Material"]

        # Filter only available features for each stage
        available_s4_num = [
            f for f in features_s4_numeric if f in df_stage4_clean.columns
        ]
        available_s4_cat = [
            f for f in features_s4_categorical if f in df_stage4_clean.columns
        ]

        available_s8_num = [
            f for f in features_s8_numeric if f in df_stage8_clean.columns
        ]
        available_s8_cat = [
            f for f in features_s8_categorical if f in df_stage8_clean.columns
        ]

        self.features_stage4 = available_s4_num + available_s4_cat
        self.features_stage8 = available_s8_num + available_s8_cat

        self.logger.info(f"\n🎯 Features Stage 4: {self.features_stage4}")
        self.logger.info(f"🎯 Features Stage 8: {self.features_stage8}")

        # Prepare Stage 4 data
        X_stage4 = df_stage4_clean[self.features_stage4]
        y_stage4 = df_stage4_clean["FuelConsumed"]
        X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(
            X_stage4, y_stage4, test_size=0.2, random_state=42
        )

        # Prepare Stage 8 data
        X_stage8 = df_stage8_clean[self.features_stage8]
        y_stage8 = df_stage8_clean["FuelConsumed"]
        X_train_8, X_test_8, y_train_8, y_test_8 = train_test_split(
            X_stage8, y_stage8, test_size=0.2, random_state=42
        )

        return {
            "stage4": (X_train_4, X_test_4, y_train_4, y_test_4),
            "stage8": (X_train_8, X_test_8, y_train_8, y_test_8),
        }

    def _calculate_metrics(self, y_true, y_pred):
        """Helper to calculate performance metrics"""
        y_pred_non_neg = np.maximum(y_pred, 0.1)
        y_true_non_neg = np.maximum(y_true, 0.1)

        try:
            rmsle = np.sqrt(mean_squared_log_error(y_true_non_neg, y_pred_non_neg))
        except ValueError:
            rmsle = float("inf")

        mape_safe = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.1))) * 100

        return {
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE_Safe": mape_safe,
            "MedianAE": median_absolute_error(y_true, y_pred),
            "RMSLE": rmsle,
            "ExplainedVar": explained_variance_score(y_true, y_pred),
        }

    def train(self):
        """
        Train both specialized XGBoost models (Stage 4 and Stage 8).
        Returns metrics for both models.
        """
        data_splits = self.prepare_data()

        # ============ TRAIN MODEL STAGE 4 ============
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🚛 ENTRENANDO MODELO PARA STAGE 4 (Camión Vacío)")
        self.logger.info("=" * 70)

        X_train_4, X_test_4, y_train_4, y_test_4 = data_splits["stage4"]

        self.logger.info(
            f"📦 Datos Stage 4 - Train: {len(X_train_4)}, Test: {len(X_test_4)}"
        )

        self.model_stage4.fit(
            X_train_4,
            y_train_4,
            eval_set=[(X_train_4, y_train_4), (X_test_4, y_test_4)],
            verbose=50,
        )

        self.logger.info(
            f"✅ Mejor iteración Stage 4: {self.model_stage4.best_iteration}"
        )

        y_pred_4 = self.model_stage4.predict(X_test_4)
        metrics_4 = self._calculate_metrics(y_test_4, y_pred_4)

        self.logger.info(f"\n📊 Métricas Stage 4 (Camión Vacío):")
        self.logger.info(f"  R²: {metrics_4['R2']:.4f}")
        self.logger.info(f"  MAE: {metrics_4['MAE']:.2f} litros")
        self.logger.info(f"  RMSE: {metrics_4['RMSE']:.2f} litros")
        self.logger.info(f"  MAPE: {metrics_4['MAPE_Safe']:.2f}%")

        # ============ TRAIN MODEL STAGE 8 ============
        self.logger.info("\n" + "=" * 70)
        self.logger.info("🚛 ENTRENANDO MODELO PARA STAGE 8 (Camión Lleno)")
        self.logger.info("=" * 70)

        X_train_8, X_test_8, y_train_8, y_test_8 = data_splits["stage8"]

        self.logger.info(
            f"📦 Datos Stage 8 - Train: {len(X_train_8)}, Test: {len(X_test_8)}"
        )

        self.model_stage8.fit(
            X_train_8,
            y_train_8,
            eval_set=[(X_train_8, y_train_8), (X_test_8, y_test_8)],
            verbose=50,
        )

        self.logger.info(
            f"✅ Mejor iteración Stage 8: {self.model_stage8.best_iteration}"
        )

        y_pred_8 = self.model_stage8.predict(X_test_8)
        metrics_8 = self._calculate_metrics(y_test_8, y_pred_8)

        self.logger.info(f"\n📊 Métricas Stage 8 (Camión Lleno):")
        self.logger.info(f"  R²: {metrics_8['R2']:.4f}")
        self.logger.info(f"  MAE: {metrics_8['MAE']:.2f} litros")
        self.logger.info(f"  RMSE: {metrics_8['RMSE']:.2f} litros")
        self.logger.info(f"  MAPE: {metrics_8['MAPE_Safe']:.2f}%")

        # Store test data for later visualization
        self.test_data = {
            "stage4": {"X": X_test_4, "y_true": y_test_4, "y_pred": y_pred_4},
            "stage8": {"X": X_test_8, "y_true": y_test_8, "y_pred": y_pred_8},
        }

        # Summary comparison
        self.logger.info("\n" + "=" * 70)
        self.logger.info("📊 COMPARACIÓN DE MODELOS")
        self.logger.info("=" * 70)
        self.logger.info(
            f"{'Métrica':<15} {'Stage 4':>15} {'Stage 8':>15} {'Mejora':>15}"
        )
        self.logger.info("-" * 70)
        for metric in ["R2", "MAE", "RMSE"]:
            val_4 = metrics_4[metric]
            val_8 = metrics_8[metric]
            if metric == "R2":
                mejora = ((val_8 - val_4) / abs(val_4)) * 100
            else:
                mejora = ((val_4 - val_8) / val_4) * 100
            self.logger.info(
                f"{metric:<15} {val_4:>15.4f} {val_8:>15.4f} {mejora:>14.2f}%"
            )

        results = {
            "stage4": {
                "samples": {"train": len(X_train_4), "test": len(X_test_4)},
                "metrics": metrics_4,
                "features": self.features_stage4,
            },
            "stage8": {
                "samples": {"train": len(X_train_8), "test": len(X_test_8)},
                "metrics": metrics_8,
                "features": self.features_stage8,
            },
            "total_consumed_fuel": self.total_consumed,
        }

        self.logger.info("\n✅ Entrenamiento completado para ambos modelos.")
        return results

    def get_predictions(self) -> pl.DataFrame:
        """
        Generate predictions using the appropriate specialized model.
        Stage 4 records use model_stage4, Stage 8 records use model_stage8.
        """
        if self.model_stage4 is None or self.model_stage8 is None:
            raise RuntimeError(
                "❌ Los modelos no están entrenados. Ejecute train() primero."
            )

        self.logger.info("🔮 Generando predicciones con modelos especializados...")

        df_all = self.cycles_data.to_pandas()
        predictions = np.zeros(len(df_all))

        # PREDICT STAGE 4 (Empty Truck)
        mask_4 = df_all["StageSequence"] == 4
        if mask_4.any():
            X_stage4 = df_all.loc[mask_4, self.features_stage4].copy()

            # Convert categorical variables
            for cat_col in self.categorical_vars:
                if cat_col in X_stage4.columns:
                    X_stage4[cat_col] = X_stage4[cat_col].astype("category")

            predictions[mask_4] = self.model_stage4.predict(X_stage4)
            self.logger.info(f"  ✅ Stage 4: {mask_4.sum()} predicciones")

        # PREDICT STAGE 8 (Loaded Truck)
        mask_8 = df_all["StageSequence"] == 8
        if mask_8.any():
            X_stage8 = df_all.loc[mask_8, self.features_stage8].copy()

            # Convert categorical variables
            for cat_col in self.categorical_vars:
                if cat_col in X_stage8.columns:
                    X_stage8[cat_col] = X_stage8[cat_col].astype("category")

            predictions[mask_8] = self.model_stage8.predict(X_stage8)
            self.logger.info(f"  ✅ Stage 8: {mask_8.sum()} predicciones")

        # Calculate residuals
        y_true_all = df_all["FuelConsumed"].values
        residuals_all = y_true_all - predictions

        # Outlier detection using corresponding models
        is_inlier = np.zeros(len(df_all), dtype=bool)
        is_outlier = np.zeros(len(df_all), dtype=bool)

        for stage_value, stage_mask in [(False, mask_4), (True, mask_8)]:
            if not stage_mask.any():
                continue

            X_numeric_stage = df_all.loc[stage_mask, self.numeric_predictor_vars]
            y_stage = df_all.loc[stage_mask, "FuelConsumed"].values
            X_with_target = np.hstack([X_numeric_stage.values, y_stage.reshape(-1, 1)])

            iso_model = (
                self.iso_forest_stage_4 if not stage_value else self.iso_forest_stage_8
            )
            outlier_preds = iso_model.predict(X_with_target)

            stage_indices = np.where(stage_mask)[0]
            is_inlier[stage_indices] = outlier_preds == 1
            is_outlier[stage_indices] = outlier_preds == -1

        # Create result DataFrame
        result = self.cycles_data.with_columns(
            [
                pl.Series("PredictedFuelXGBoost", predictions),
                pl.Series("residual", residuals_all),
                pl.Series("is_inlier", is_inlier),
                pl.Series("is_outlier", is_outlier),
            ]
        )

        self.logger.info(
            "✅ Predicciones generadas exitosamente usando modelos especializados."
        )
        # use -> .filter((pl.col("PredictedFuelXGBoost") > 0) & (pl.col("FuelConsumed") > 0))
        return result.select(
            [
                "cycle_group",
                "TimeStampIni",
                "TimeStampFin",
                "ShiftDate",
                "Equipment",
                "TruckFleet",
                "SpeedAvg",
                "AvgSlopePercent",
                "AvgAcceleration",
                "TimeEfficiencyPercentage",
                "Latitude",
                "Longitude",
                "Elevation",
                "TotalMeasuredTonnage",
                "Distance",
                "StageSequence",
                "Destination",
                "DestinationType",
                "Material",
                "Shovel",
                "RecordsInCycle",
                "FuelConsumed",
                "CycleDurationSeconds",
                "is_inlier",
                "is_outlier",
                "PredictedFuelXGBoost",
                "residual",
            ]
        ).sort("cycle_group")
