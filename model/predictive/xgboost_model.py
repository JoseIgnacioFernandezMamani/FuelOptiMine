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
            f"  ✂️ {stage_name}: Eliminados {n_outliers} outliers ({outlier_pct:.2f}%)"
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

    def plot_predictions(self, save_path: str = None, results: dict = None):
        """
        Generate visual plots comparing predictions vs actual values for BOTH stages.
        Creates separate visualizations for Stage 4 and Stage 8.
        """
        if not self.test_data:
            self.logger.error("❌ No hay datos de test. Entrene el modelo primero.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        plt.style.use("default")

        stages = [
            ("stage4", "Stage 4 (Vacío)", "blue"),
            ("stage8", "Stage 8 (Lleno)", "red"),
        ]

        for idx, (stage_key, stage_name, color) in enumerate(stages):
            data = self.test_data[stage_key]
            y_test = np.array(data["y_true"])
            y_pred = np.array(data["y_pred"])

            if results and stage_key in results and "metrics" in results[stage_key]:
                r2 = results[stage_key]["metrics"]["R2"]
                rmse = results[stage_key]["metrics"]["RMSE"]
            else:
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Plot 1: Time series
            indices = range(1, len(y_test) + 1)
            axes[idx, 0].plot(
                indices,
                y_test,
                "o-",
                alpha=0.7,
                label="Real",
                markersize=3,
                linewidth=1,
            )
            axes[idx, 0].plot(
                indices,
                y_pred,
                "o-",
                alpha=0.7,
                label="Predicción",
                markersize=3,
                linewidth=1,
                color=color,
            )
            axes[idx, 0].set_xlabel("Índice")
            axes[idx, 0].set_ylabel("Combustible (L)")
            axes[idx, 0].set_title(f"{stage_name} - Serie Temporal")
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)

            # Plot 2: Scatter
            axes[idx, 1].scatter(y_test, y_pred, alpha=0.6, s=30, color=color)
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            axes[idx, 1].plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=2,
                label="Perfecto",
            )
            axes[idx, 1].set_xlabel("Real (L)")
            axes[idx, 1].set_ylabel("Predicción (L)")
            axes[idx, 1].set_title(f"{stage_name} - Scatter Plot")
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].text(
                0.05,
                0.95,
                f"R² = {r2:.3f}\nRMSE = {rmse:.2f}",
                transform=axes[idx, 1].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Plot 3: Residuals
            residuals = y_test - y_pred
            axes[idx, 2].scatter(y_pred, residuals, alpha=0.6, s=30, color=color)
            axes[idx, 2].axhline(y=0, color="k", linestyle="--", linewidth=2)
            axes[idx, 2].set_xlabel("Predicción (L)")
            axes[idx, 2].set_ylabel("Residual (L)")
            axes[idx, 2].set_title(f"{stage_name} - Residuales")
            axes[idx, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"💾 Gráfica guardada en: {save_path}")
            except Exception as e:
                self.logger.error(f"❌ Error guardando gráfica: {e}")

        plt.show()

    def get_feature_importance(self, stage="both"):
        """
        Extract feature importance from trained models.

        Args:
            stage: "stage4", "stage8", or "both"
        """
        importance_data = {}

        if stage in ["stage4", "both"] and self.model_stage4 is not None:
            importance_4 = self.model_stage4.feature_importances_
            importance_data["stage4"] = pd.DataFrame(
                {"feature": self.features_stage4, "importance": importance_4}
            ).sort_values("importance", ascending=False)

        if stage in ["stage8", "both"] and self.model_stage8 is not None:
            importance_8 = self.model_stage8.feature_importances_
            importance_data["stage8"] = pd.DataFrame(
                {"feature": self.features_stage8, "importance": importance_8}
            ).sort_values("importance", ascending=False)

        return importance_data

    def print_feature_importance(self):
        """Print feature importance for both models."""
        importance_dict = self.get_feature_importance(stage="both")

        self.logger.info("\n" + "=" * 70)
        self.logger.info("📊 IMPORTANCIA DE CARACTERÍSTICAS")
        self.logger.info("=" * 70)

        for stage_key, stage_name in [
            ("stage4", "Stage 4 (Vacío)"),
            ("stage8", "Stage 8 (Lleno)"),
        ]:
            if stage_key in importance_dict:
                self.logger.info(f"\n🔍 {stage_name}:")
                df = importance_dict[stage_key]
                for _, row in df.head(10).iterrows():
                    self.logger.info(f"  {row['feature']:<30} {row['importance']:.4f}")

    def save_model(self, base_path: str = "xgboost_dual"):
        """Save both trained models in JSON format."""
        path_4 = f"{base_path}_stage4.json"
        path_8 = f"{base_path}_stage8.json"

        self.model_stage4.get_booster().save_model(path_4)
        self.model_stage8.get_booster().save_model(path_8)

        self.logger.info(f"💾 Modelo Stage 4 guardado en: {path_4}")
        self.logger.info(f"💾 Modelo Stage 8 guardado en: {path_8}")
        self.logger.info(
            "⚠️  IMPORTANTE: Usar formato JSON para preservar información categórica"
        )

    def load_model(self, base_path: str = "xgboost_dual"):
        """Load both trained models from JSON format."""
        path_4 = f"{base_path}_stage4.json"
        path_8 = f"{base_path}_stage8.json"

        self.model_stage4.load_model(path_4)
        self.model_stage8.load_model(path_8)

        self.logger.info(f"✅ Modelo Stage 4 cargado desde: {path_4}")
        self.logger.info(f"✅ Modelo Stage 8 cargado desde: {path_8}")

    def explain_model(self, stage="both", max_display: int = 15, plots: bool = False):
        """
        Generate SHAP-based explanations for the trained models.

        Args:
            stage: "stage4", "stage8", or "both"
            max_display: Maximum number of features to display
            plots: Whether to show SHAP plots
        """
        stages_to_explain = []
        if stage in ["stage4", "both"]:
            stages_to_explain.append(("stage4", self.model_stage4, "Stage 4 (Vacío)"))
        if stage in ["stage8", "both"]:
            stages_to_explain.append(("stage8", self.model_stage8, "Stage 8 (Lleno)"))

        for stage_key, model, stage_name in stages_to_explain:
            if stage_key not in self.test_data:
                self.logger.warning(f"⚠️  No hay datos de test para {stage_name}")
                continue

            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"🔍 Explicaciones SHAP para {stage_name}")
            self.logger.info(f"{'='*70}")

            X_test = self.test_data[stage_key]["X"]

            try:
                explainer = shap.TreeExplainer(
                    model, feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(X_test, check_additivity=False)

                shap_values_explanation = shap.Explanation(
                    values=shap_values,
                    base_values=explainer.expected_value,
                    data=X_test,
                    feature_names=list(X_test.columns),
                )

                shap_importance = np.abs(shap_values).mean(axis=0)
                importance_df = pd.DataFrame(
                    {"feature": list(X_test.columns), "mean_abs_shap": shap_importance}
                ).sort_values("mean_abs_shap", ascending=False)

                self.logger.info(f"\n📊 Top {max_display} variables por SHAP:")
                for _, row in importance_df.head(max_display).iterrows():
                    self.logger.info(
                        f"  {row['feature']:<30} {row['mean_abs_shap']:.4f}"
                    )

                # Validate additivity
                shap_sum = shap_values.sum(axis=1) + explainer.expected_value
                preds = model.predict(X_test)
                additivity_error = np.abs(shap_sum - preds) / (np.abs(preds) + 1e-8)

                self.logger.info(f"\n📈 Error de aditividad SHAP:")
                self.logger.info(f"  Media: {additivity_error.mean():.6f}")
                self.logger.info(f"  Máximo: {additivity_error.max():.6f}")

                if plots:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                    plt.sca(ax1)
                    shap.summary_plot(
                        shap_values_explanation.values,
                        X_test,
                        feature_names=list(X_test.columns),
                        show=False,
                    )
                    ax1.set_title(f"SHAP Summary - {stage_name}")

                    shap.plots.bar(shap_values_explanation, ax=ax2, show=False)
                    ax2.set_title(f"SHAP Feature Importance - {stage_name}")

                    plt.tight_layout()
                    plt.show()

                self.logger.info(f"✅ Explicaciones SHAP calculadas para {stage_name}")

            except Exception as e:
                self.logger.error(
                    f"❌ Error calculando SHAP para {stage_name}: {str(e)}"
                )

    def predict_manual(self, stage: int = None):
        """
        Perform manual prediction by entering feature values via console.

        Args:
            stage: 4 or 8. If None, will ask user to select.
        """
        if self.model_stage4 is None or self.model_stage8 is None:
            self.logger.error("❌ Modelos no entrenados. Ejecute train() primero.")
            return

        # Ask for stage if not provided
        if stage is None:
            while True:
                try:
                    stage = int(
                        input("\n¿Qué stage desea predecir? (4 = vacío, 8 = lleno): ")
                    )
                    if stage in [4, 8]:
                        break
                    print("Por favor ingrese 4 u 8")
                except ValueError:
                    print("Por favor ingrese un número válido")

        # Select appropriate model and features
        if stage == 4:
            model = self.model_stage4
            features = self.features_stage4
            stage_name = "Stage 4 (Camión Vacío)"
        else:
            model = self.model_stage8
            features = self.features_stage8
            stage_name = "Stage 8 (Camión Lleno)"

        self.logger.info(f"\n🔮 Predicción para {stage_name}")
        self.logger.info(f"Ingrese los valores para las siguientes variables:\n")

        input_data = {}

        # Collect numeric features
        for var in features:
            if var in self.numeric_predictor_vars:
                while True:
                    try:
                        value = float(input(f"  {var}: "))
                        input_data[var] = value
                        break
                    except ValueError:
                        print("    ⚠️  Por favor, ingrese un número válido")

        # Collect categorical features
        for var in features:
            if var in self.categorical_vars:
                value = input(f"  {var}: ")
                input_data[var] = value

        # Build DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert categorical columns
        for cat_col in self.categorical_vars:
            if cat_col in input_df.columns:
                input_df[cat_col] = input_df[cat_col].astype("category")

        # Make prediction
        prediction = model.predict(input_df)

        self.logger.info(f"\n✅ Predicción de consumo: {prediction[0]:.2f} litros")
        return prediction[0]

    def compare_models_performance(self, results: dict):
        """
        Generate a comprehensive comparison report between Stage 4 and Stage 8 models.
        """
        if "stage4" not in results or "stage8" not in results:
            self.logger.error("❌ Resultados incompletos para comparación")
            return

        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 REPORTE COMPARATIVO DE MODELOS")
        self.logger.info("=" * 80)

        metrics_4 = results["stage4"]["metrics"]
        metrics_8 = results["stage8"]["metrics"]
        samples_4 = results["stage4"]["samples"]
        samples_8 = results["stage8"]["samples"]

        # Data distribution
        self.logger.info("\n📦 Distribución de Datos:")
        self.logger.info(
            f"  Stage 4: {samples_4['train']} train, {samples_4['test']} test"
        )
        self.logger.info(
            f"  Stage 8: {samples_8['train']} train, {samples_8['test']} test"
        )

        # Metrics comparison
        self.logger.info("\n📈 Comparación de Métricas:")
        self.logger.info(
            f"{'Métrica':<20} {'Stage 4':>15} {'Stage 8':>15} {'Diferencia':>15}"
        )
        self.logger.info("-" * 80)

        for metric in ["R2", "MAE", "RMSE", "MAPE_Safe", "MedianAE"]:
            val_4 = metrics_4[metric]
            val_8 = metrics_8[metric]
            diff = val_8 - val_4

            if metric == "R2":
                better = "✅ Stage 8" if val_8 > val_4 else "✅ Stage 4"
            else:
                better = "✅ Stage 4" if val_4 < val_8 else "✅ Stage 8"

            self.logger.info(
                f"{metric:<20} {val_4:>15.4f} {val_8:>15.4f} {diff:>15.4f} {better}"
            )

        # Features comparison
        self.logger.info("\n🔍 Comparación de Features:")
        self.logger.info(f"  Stage 4 features: {len(self.features_stage4)}")
        self.logger.info(f"  Stage 8 features: {len(self.features_stage8)}")

        common_features = set(self.features_stage4) & set(self.features_stage8)
        unique_4 = set(self.features_stage4) - set(self.features_stage8)
        unique_8 = set(self.features_stage8) - set(self.features_stage4)

        self.logger.info(
            f"\n  Comunes: {len(common_features)} - {list(common_features)}"
        )
        self.logger.info(f"  Únicas Stage 4: {len(unique_4)} - {list(unique_4)}")
        self.logger.info(f"  Únicas Stage 8: {len(unique_8)} - {list(unique_8)}")

        # Best iteration comparison
        self.logger.info("\n⚙️  Convergencia:")
        self.logger.info(
            f"  Stage 4 best iteration: {self.model_stage4.best_iteration}"
        )
        self.logger.info(
            f"  Stage 8 best iteration: {self.model_stage8.best_iteration}"
        )

        self.logger.info("\n" + "=" * 80)
