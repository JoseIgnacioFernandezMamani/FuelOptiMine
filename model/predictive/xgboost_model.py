import polars as pl
import numpy as np
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
    Class for training an XGBoost model using native support for categorical variables.
    More efficient than manual preprocessing (OneHot/Binary encoding).
    """

    def __init__(
        self,
        numeric_predictor_vars: List[str],
        categorical_vars: List[str],
        max_cat_to_onehot: int = 4,  # XGBoost will automatically decide between one-hot and partitioning
    ):
        """
        Initialize model with native categorical support.

        Args:
            numeric_predictor_vars: Lista de variables numéricas
            categorical_vars: Lista de variables categóricas
            max_cat_to_onehot: threshold to decide between one-hot and partitioning
        """
        self.categorical_vars = categorical_vars
        self.numeric_predictor_vars = numeric_predictor_vars
        self.df = pl.DataFrame()
        self.cycles_data = pl.DataFrame()

        # XGBoost Regressor with categorical support
        self.model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",  # Requerido para categorical support
            device="cuda",
            random_state=42,
            eval_metric="mae",
            early_stopping_rounds=100,
            reg_alpha=0.01,
            reg_lambda=1,
            verbosity=1,
            # ✨ Parámetros clave para soporte nativo de categóricas
            enable_categorical=True,
            max_cat_to_onehot=max_cat_to_onehot,  # Control automático de encoding
        )

        # Results and testing data
        self.y_test = None
        self.y_pred = None
        self.feature_names = []

        # outliers detection model
        self.iso_forest_stage_4 = None
        self.iso_forest_stage_8 = None

        # store test data
        self.X_test = None

        # self.logger
        self.logger = get_logger("XGBoost", "xgb.log", console=True)

        # rmse regresion for desicion making
        self.rmse_threshold = 0

    def load_data(self, csv_path: str = "unified_data_T-210.csv"):
        """
        hardcoding for moment, should be improved
        """
        model = LinearRegressionModel()
        results = model.train_models()
        self.rmse_threshold = (
            results["stage_4"]["train_metrics"]["rmse"]
            + results["stage_8"]["train_metrics"]["rmse"]
        ) / 2
        self.df = model.get_predictions()

    def transform_cycles_data(self):
        """
        Process data to identify cycles and calculate fuel consumption metrics.
        """
        self.logger.info("Transformando datos de cycle")

        df = self.df
        result = df.with_columns(
            [
                # Calculation of fuel consumption, with default values
                pl.when(
                    (
                        (
                            pl.col("FuelConsumed") - pl.col("PredictedFuelConsumption")
                        ).abs()
                        > self.rmse_threshold
                    )
                    & (pl.col("PredictedFuelConsumption") > 0)
                )
                .then(pl.col("PredictedFuelConsumption"))
                .otherwise(pl.col("FuelConsumed"))
                .alias("FuelConsumed"),
                pl.when(pl.col("StageSequence") == 8)
                .then(True)
                .otherwise(False)
                .alias("StageSequence"),
            ]
        )

        # clean columns
        cols_to_clean = self.numeric_predictor_vars + [
            "FuelConsumed"
        ]  # special case: fuelconsumed, target variable
        cols_to_clean.remove("StageSequence")  # special case: stage secuence bool

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

    def prepare_data(self):
        """
        Prepare data for XGBoost training using native categorical support.
        """
        self.logger.info("Preparing data for training...")

        # convert to pandas
        df = self.cycles_data.to_pandas().copy()

        # Split numeric and categorical features
        X_numeric = df[self.numeric_predictor_vars]
        X_categorical = df[self.categorical_vars]
        y = df["FuelConsumed"]

        # Assign categorical dtype
        for cat_col in self.categorical_vars:
            X_categorical[cat_col] = X_categorical[cat_col].astype("category")
            print(
                f"Variable '{cat_col}' convertida a categoría con {X_categorical[cat_col].nunique()} valores únicos"
            )

        # convine numeric and categorical features
        df_full = pd.concat([X_numeric, X_categorical], axis=1)
        df_full["FuelConsumed"] = y
        df_full["StageSequence"] = df["StageSequence"]

        # List to hold cleaned data per stage
        df_clean_list = []

        # Apply Isolation Forest separately for each StageSequence value
        for stage_value in [True, False]:
            df_stage = df_full[df_full["StageSequence"] == stage_value].copy()
            if df_stage.empty:
                continue

            self.logger.info(
                f"Processing StageSequence={stage_value} with {len(df_stage)} records"
            )

            # Prepare features for Isolation Forest: numeric + target
            X_stage_iso = df_stage[self.numeric_predictor_vars].values
            y_stage = df_stage["FuelConsumed"].values
            X_with_target = np.hstack([X_stage_iso, y_stage.reshape(-1, 1)])

            # Fit Isolation Forest
            iso_forest_stage = IsolationForest(
                contamination=0.05,
                random_state=42,
                n_estimators=200,
                max_samples="auto",
            )
            iso_forest_stage.fit(X_with_target)
            mask = iso_forest_stage.predict(X_with_target) == 1

            n_outliers = np.sum(~mask)
            outlier_pct = (n_outliers / len(df_stage)) * 100
            self.logger.info(
                f"StageSequence={stage_value}: Removed {n_outliers} outliers ({outlier_pct:.2f}%)"
            )

            # Keep only inliers
            df_stage_clean = df_stage[mask].copy()
            df_clean_list.append(df_stage_clean)

            if stage_value:
                self.iso_forest_stage_8 = iso_forest_stage
            else:
                self.iso_forest_stage_4 = iso_forest_stage

        # Concatenate cleaned stages
        df_cleaned = pd.concat(df_clean_list, axis=0).reset_index(drop=True)

        # Split back numeric, categorical, and target
        X_numeric_clean = df_cleaned[self.numeric_predictor_vars]
        X_categorical_clean = df_cleaned[self.categorical_vars]
        y_clean = df_cleaned["FuelConsumed"]

        # Final feature matrix for XGBoost
        X_final = pd.concat([X_numeric_clean, X_categorical_clean], axis=1)

        # Save feature names
        self.feature_names = list(X_final.columns)

        self.logger.info(
            f"Final dataset size after stage-wise outlier removal: {len(X_final)} records"
        )
        self.logger.info(f"Numeric features: {self.numeric_predictor_vars}")
        self.logger.info(f"Categorical features: {self.categorical_vars}")
        self.logger.info("Final dtypes:")
        for col in X_final.columns:
            self.logger.info(f"  {col}: {X_final[col].dtype}")

        return train_test_split(X_final, y_clean, test_size=0.2, random_state=42)

    def train(self):
        """
        Train the XGBoost model with native categorical support.

        This method handles the complete training pipeline:
        - Prepares training and testing datasets (numeric + categorical features).
        - Trains an XGBoost model with native categorical handling.
        - Calculates multiple performance metrics (R², MAE, RMSE, RMSLE, MAPE, etc.).
        - Stores test predictions for later visualization and analysis.

        Logging:
            - Logs dataset size and dimensionality.
            - Logs model training progress, best iteration, and feature types.
            - Logs training completion with metrics.

        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics including:
                - R2: Coefficient of determination
                - MAE: Mean Absolute Error
                - RMSE: Root Mean Squared Error
                - MAPE_Safe: Mean Absolute Percentage Error (safe version)
                - MedianAE: Median Absolute Error
                - RMSLE: Root Mean Squared Logarithmic Error
                - ExplainedVar: Explained Variance Score
        """
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Save original test indices for traceability
        self.test_indices_original = X_test.index

        self.logger.info(f"Entrenando con {len(X_train)} records...")

        # train model and evaluate simultaneously
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50,
        )

        self.logger.info(f"Mejor iteración: {self.model.best_iteration}")

        # Mostrar información sobre cómo XGBoost procesó las categóricas
        self.logger.info(
            f"Tipos de features detectados por XGBoost: {self.model.get_booster().feature_types}"
        )

        # make predictionson test set
        y_pred = self.model.predict(X_test)

        # Ensure predictions are non-negative (important for RMSLE and fuel data)
        y_pred_non_negative = np.maximum(y_pred, 0.1)
        y_test_non_negative = np.maximum(y_test, 0.1)

        try:
            rmsle = np.sqrt(
                mean_squared_log_error(y_test_non_negative, y_pred_non_negative)
            )
        except ValueError:
            # If RMSLE cannot be computed (e.g., due to invalid values)
            rmsle = float("inf")

        # save results
        self.y_test = y_test
        self.y_pred = y_pred
        self.X_test = X_test

        # MAPE safe calculation to avoid division by zero
        mape_safe = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.1))) * 100

        metrics = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAPE_Safe": mape_safe,
            "MedianAE": median_absolute_error(y_test, y_pred),
            "RMSLE": rmsle,
            "ExplainedVar": explained_variance_score(y_test, y_pred),
        }

        # store evaluation metrics
        results = {
            "model": str(type(self.model).__name__),
            # "isolation_forest": str(type(self.iso_forest).__name__),
            "samples": {
                "train": len(X_train),
                "test": len(X_test),
            },
            "metrics": metrics,
            "multicollinearity": analyze_multicollinearity(
                self.cycles_data, self.numeric_predictor_vars
            ),
        }
        log_results(self.numeric_predictor_vars, "all", results, self.logger)
        self.logger.info(
            "Training completed. Evaluation metrics calculated successfully."
        )
        return results

    def plot_predictions(self, save_path: str, results: dict):
        """
        Generate visual plots comparing model predictions against actual values.
        This method creates two visualizations:
        1. Time series plot showing the progression of predictions vs real values.
        2. Scatter plot comparing predictions against real values with a "perfect fit" line.
        Metrics (R² and RMSE) are displayed on the scatter plot for quick interpretation.

        Args:
            save_path (str, optional): File path to save the generated plots.
                                    If None, plots are only displayed.
            results (dict, optional): Results dictionary containing metrics.
                                    If None, will calculate metrics from y_test and y_pred.

        Logging:
            - Logs error if model is not trained before plotting.
            - Logs confirmation if plots are saved to disk.
        """
        if self.y_test is None or self.y_pred is None:
            self.logger.error("Model must be trained before plotting predictions.")
            return

        # Validar que tenemos datos válidos
        if len(self.y_test) == 0 or len(self.y_pred) == 0:
            self.logger.error("No hay datos de predicción para graficar.")
            return

        # Obtener métricas
        try:
            if results and "metrics" in results:
                # Estructura anidada: results["metrics"]["R2"]
                r2 = results["metrics"]["R2"]
                rmse = results["metrics"]["RMSE"]
            elif results and "R2" in results:
                # Estructura plana: results["R2"]
                r2 = results["R2"]
                rmse = results["RMSE"]
            else:
                # Calcular métricas si no se proporcionan
                from sklearn.metrics import r2_score, mean_squared_error

                r2 = r2_score(self.y_test, self.y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
                self.logger.info(
                    "Métricas calculadas automáticamente ya que no se proporcionaron en results."
                )
        except KeyError as e:
            self.logger.error(f"Error accediendo a las métricas: {e}")
            # Calcular métricas como fallback
            from sklearn.metrics import r2_score, mean_squared_error

            r2 = r2_score(self.y_test, self.y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))

        plt.style.use("default")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

        # Convertir a numpy arrays si es necesario para evitar problemas de indexing
        y_test_array = (
            np.array(self.y_test) if hasattr(self.y_test, "__iter__") else self.y_test
        )
        y_pred_array = (
            np.array(self.y_pred) if hasattr(self.y_pred, "__iter__") else self.y_pred
        )

        indices = range(1, len(y_test_array) + 1)

        # Gráfica 1: Series de tiempo
        ax1.plot(
            indices,
            y_test_array,
            "o-",
            color="blue",
            alpha=0.7,
            label="Valores Reales",
            markersize=4,
            linewidth=1,
        )
        ax1.plot(
            indices,
            y_pred_array,
            "o-",
            color="red",
            alpha=0.7,
            label="Predicciones",
            markersize=4,
            linewidth=1,
        )
        ax1.set_xlabel("Índice del Registro")
        ax1.set_ylabel("Consumo de Combustible (Litros)")
        ax1.set_title("Predicciones vs Valores Reales (XGBoost Nativo)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfica 2: Scatter plot
        ax2.scatter(y_test_array, y_pred_array, alpha=0.6, color="green", s=30)

        # Calcular rango para la línea de predicción perfecta
        min_val = min(np.min(y_test_array), np.min(y_pred_array))
        max_val = max(np.max(y_test_array), np.max(y_pred_array))

        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            label="Predicción Perfecta",
            linewidth=2,
        )
        ax2.set_xlabel("Valores Reales (Litros)")
        ax2.set_ylabel("Predicciones (Litros)")
        ax2.set_title("Scatter Plot: XGBoost con Soporte Nativo")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Añadir métricas al gráfico
        ax2.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}\nRMSE = {rmse:.2f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Gráfica 3: Matriz de correlación
        if results and "multicollinearity" in results:
            try:
                corr_matrix = results["multicollinearity"]["correlation_matrix"]
                corr_df = pd.DataFrame(corr_matrix)
                sns.heatmap(
                    corr_df, annot=False, cmap="coolwarm", center=0, cbar=True, ax=ax3
                )
                ax3.set_title("Matriz de Correlación (Predictoras)")
            except Exception as e:
                self.logger.error(f"Error graficando matriz de correlación: {e}")
                ax3.text(
                    0.5,
                    0.5,
                    "No se pudo graficar\nmatriz de correlación",
                    ha="center",
                    va="center",
                )
        else:
            ax3.text(
                0.5,
                0.5,
                "No hay matriz de correlación disponible",
                ha="center",
                va="center",
            )

        plt.tight_layout()

        # Guardar si se especifica ruta
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                self.logger.info(f"Gráfica guardada en: {save_path}")
            except Exception as e:
                self.logger.error(f"Error guardando la gráfica: {e}")

        plt.show()

    def get_feature_importance(self):
        """
        Extract and return feature importance from trained model.

        Feature importance is computed directly from the trained XGBoost model
        using gain-based importance scores.

        Returns:
            pd.DataFrame: Sorted DataFrame with two columns:
                - feature: Feature name
                - importance: Importance score (higher = more important)

        Logging:
            - Logs confirmation when feature importance has been extracted.
        """
        if self.model is None:
            self.logger.error(
                "Model must be trained before extracting feature importance."
            )
            return pd.DataFrame()

        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        self.logger.info("Feature importance extracted successfully.")

        return feature_importance

    def print_feature_importance(self):
        """
        Print feature importance grouped by feature type (numeric vs categorical).

        Uses the feature importance extracted by `get_feature_importance()` and
        separates them into numeric and categorical groups for readability.

        Logging:
            - Logs the importance of numeric and categorical variables separately.
        """
        importance_df = self.get_feature_importance()
        if importance_df.empty:
            self.logger.error("No feature importance data available.")
            return

        self.logger.info(
            "\n===== Importancia de Características (XGBoost Nativo) ====="
        )

        # split by type
        numeric_features = importance_df[
            importance_df["feature"].isin(self.numeric_predictor_vars)
        ]
        categorical_features = importance_df[
            importance_df["feature"].isin(self.categorical_vars)
        ]

        self.logger.info("\n Variables Numéricas:")
        for _, row in numeric_features.iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        self.logger.info("\n Variables Categóricas (Procesamiento Nativo):")
        for _, row in categorical_features.iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        self.logger.info(f"\nResumen:")
        self.logger.info(f"  Total features: {len(importance_df)}")
        self.logger.info(f"  Features numéricas: {len(numeric_features)}")
        self.logger.info(f"  Features categóricas: {len(categorical_features)}")

    def save_model(self, path: str = "xgboost_native_categorical.json"):
        """
        Save trained XGBoost model in JSON format.

        JSON format is mandatory when using native categorical support,
        since binary formats (like .bin) do not preserve categorical metadata.

        Args:
            path (str, optional): File path to save the model. Defaults to JSON.

        Logging:
            - Logs confirmation of model saving and reminders about JSON usage.
        """
        self.model.get_booster().save_model(path)
        self.logger.info(f" Modelo guardado en: {path}")
        self.logger.info(
            " IMPORTANTE: Usar formato JSON para preservar información categórica"
        )

    def get_predictions(self) -> pl.DataFrame:
        """
        Generate extended dataset with predictions, residuals, and outlier detection.

        This method applies the trained model to the **entire dataset** (`cycles_data`)
        and enriches it with:
            - Predicted fuel consumption for each cycle
            - Residuals (difference between actual and predicted values)
            - Inlier/outlier flags based on Isolation Forest applied to numeric variables

        Returns:
            pl.DataFrame: Original dataset with new columns:
                - PredictedFuelXGBoost
                - residual
                - is_inlier
                - is_outlier

        Raises:
            RuntimeError: If the model is not trained before calling this method.

        Logging:
            - Logs when predictions and residuals are added to the dataset.
        """
        # Check that the model is trained
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError(
                "El modelo no está entrenado. Ejecute primero train_model()."
            )

        # convert data to pandas for compatibility with XGBoost
        df_all = self.cycles_data.to_pandas()

        # Prepare numeric and categorical features consistently with training
        X_all = df_all[self.numeric_predictor_vars + self.categorical_vars].copy()

        # Convert categorical variables to 'category' dtype (same as in training)
        for cat_col in self.categorical_vars:
            X_all[cat_col] = X_all[cat_col].astype("category")

        # run predictions for all available data
        y_pred_all = self.model.predict(X_all)
        y_true_all = df_all["FuelConsumed"].values

        # compute residuals
        residuals_all = y_true_all - y_pred_all

        # apply outlier detection on numeric features
        # CORRECCIÓN: Verificar que los modelos de IsolationForest están entrenados
        if self.iso_forest_stage_8 is None or self.iso_forest_stage_4 is None:
            raise RuntimeError(
                "Los modelos de detección de outliers no están entrenados. Ejecute primero train()."
            )

        # CORRECCIÓN: Aplicar detección de outliers por etapa usando los modelos entrenados
        is_inlier = np.zeros(len(df_all), dtype=bool)
        is_outlier = np.zeros(len(df_all), dtype=bool)

        # Para cada etapa, usar el modelo IsolationForest correspondiente
        for stage_value in [True, False]:
            stage_mask = df_all["StageSequence"] == stage_value
            if not stage_mask.any():
                continue

            # Obtener datos numéricos para esta etapa
            X_numeric_stage = df_all.loc[stage_mask, self.numeric_predictor_vars]
            y_stage = df_all.loc[stage_mask, "FuelConsumed"].values
            X_with_target = np.hstack([X_numeric_stage.values, y_stage.reshape(-1, 1)])

            # Usar el modelo IsolationForest correspondiente
            if stage_value:
                outlier_predictions = self.iso_forest_stage_8.predict(X_with_target)
            else:
                outlier_predictions = self.iso_forest_stage_4.predict(X_with_target)

            # Actualizar las máscaras de inliers/outliers
            stage_indices = np.where(stage_mask)[0]
            is_inlier[stage_indices] = outlier_predictions == 1
            is_outlier[stage_indices] = outlier_predictions == -1

        # convert back to polars and enrich dataset
        result = self.cycles_data.with_columns(
            [
                pl.Series("PredictedFuelXGBoost", y_pred_all),
                pl.Series("residual", residuals_all),
                pl.Series("is_inlier", is_inlier),
                pl.Series("is_outlier", is_outlier),
            ]
        ).with_columns(
            pl.when(pl.col("StageSequence")).then(8).otherwise(4).alias("StageSequence")
        )

        self.logger.info("Predictions, residuals, and outlier flags added to dataset.")
        return result.select(
            [
                "cycle_group",
                "TimeStampIni",
                "TimeStampFin",
                "ShiftDate",
                "Equipment",
                "TruckFleet",
                "MedianFuelLevelLiters",
                "AvgSpeed",
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

    def explain_model(self, max_display: int = 15, plots: bool = False):
        """
        Generate SHAP-based explanations for the trained XGBoost model.
        This method uses SHAP (SHapley Additive exPlanations) to interpret the trained
        XGBoost model. SHAP values provide insights into how much each feature contributes
        to the model's predictions, both on average and per individual prediction.

        The method performs the following steps:
            1. Verifies that the model and test data exist.
            2. Constructs a SHAP TreeExplainer, tailored for tree-based models like XGBoost.
            3. Calculates SHAP values for the test dataset.
            4. Aggregates and ranks features based on their mean absolute contribution.
            5. Logs the top contributing features (limited by `max_display`).
            6. Evaluates the additivity property of SHAP explanations by comparing
            the reconstructed prediction (SHAP sum + expected value) against
            the model's raw predictions.
            7. Logs the relative mean and maximum additivity errors as a measure
            of explanation accuracy.

        Args:
            max_display (int, optional): Maximum number of top features to display.
                                        Defaults to 15.

        Returns:
            tuple: (shap_values_explanation, importance_df) for further analysis
        """
        if not hasattr(self, "model") or self.model is None:
            self.logger.error("❌ Error: El modelo no ha sido entrenado todavía.")

        if not hasattr(self, "X_test") or self.X_test is None:
            self.logger.error(
                "❌ Error: No existen datos de test. Ejecute train() primero."
            )

        self.logger.info("\n Calculando explicaciones con SHAP...")
        try:
            # Crear SHAP explainer
            explainer = shap.TreeExplainer(
                self.model, feature_perturbation="interventional"
            )

            # Compute SHAP values for test data
            shap_values = explainer.shap_values(self.X_test, check_additivity=False)

            # Wrap SHAP results into an Explanation object (for compatibility with plots)
            shap_values_explanation = shap.Explanation(
                values=shap_values,
                base_values=explainer.expected_value,
                data=self.X_test,
                feature_names=self.feature_names,
            )

            # Calculate average absolute importance per feature
            shap_importance = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame(
                {"feature": self.feature_names, "mean_abs_shap": shap_importance}
            ).sort_values("mean_abs_shap", ascending=False)

            self.logger.info("\n===== Aporte de variables según SHAP =====")
            for _, row in importance_df.head(
                max_display
            ).iterrows():  # Corregido: quitado asteriscos
                self.logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

            # Validate SHAP additivity property (prediction = sum(shap) + expected_value)
            shap_sum = shap_values.sum(axis=1) + explainer.expected_value
            preds = self.model.predict(self.X_test)
            additivity_error = np.abs(shap_sum - preds) / (np.abs(preds) + 1e-8)

            self.logger.info(
                f"📊 Media error relativo de aditividad: {additivity_error.mean():.6f}"
            )  # Corregido: formato f-string
            self.logger.info(
                f"📈 Máximo error relativo: {additivity_error.max():.6f}"
            )  # Corregido: formato f-string

            # Store SHAP values for later use
            self.shap_values = shap_values
            self.shap_explainer = explainer
            self.shap_explanation = shap_values_explanation

            self.logger.info("✅ Explicaciones SHAP calculadas correctamente.")

            if plots:
                # create combined plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
                manager = plt.get_current_fig_manager()

                plt.sca(ax1)
                shap.summary_plot(
                    shap_values_explanation.values,
                    self.X_test,
                    feature_names=self.feature_names,
                    show=False,
                )
                shap.plots.bar(shap_values_explanation, ax=ax2, show=False)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1)
                plt.tight_layout(pad=0.5)
                plt.show()

        except Exception as e:
            self.logger.error(f"❌ Error calculando explicaciones SHAP: {str(e)}")
        # Opcional: gráfico resumen
        # shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)
        # shap.plots.bar(shap_values_explanation)
        # shap.plots.waterfall(shap_values_explanation[0])

    def predict_manual(self):
        """
        Perform manual prediction by entering feature values via console input.

        This method allows interactive predictions by prompting the user to input
        values for both numeric and categorical features. It is useful for testing
        the model with custom scenarios, "what-if" analysis, or educational purposes.

        The method performs the following steps:
            1. Validates that a trained model is available.
            2. Prompts the user to input numeric feature values (validated as floats).
            3. Prompts the user to input categorical feature values as strings.
            4. Constructs a DataFrame from the collected inputs.
            5. Ensures categorical columns are properly typed.
            6. Concatenates numeric and categorical features into the final input matrix.
            7. Passes the data into the trained model for prediction.
            8. Logs the predicted fuel consumption.

        Returns:
            float: Predicted fuel consumption value for the provided inputs.
        """
        if not hasattr(self, "model") or self.model is None:
            self.logger.error(
                "❌ Error: El modelo no está entrenado. Ejecute train() primero."
            )
            return

        self.logger.info("\n Ingrese los valores para la predicción:")

        # collect numeric feature inputs
        input_data = {}
        for var in self.numeric_predictor_vars:
            while True:
                try:
                    value = float(input(f"{var}: "))
                    input_data[var] = value
                    break
                except ValueError:
                    print("Por favor, ingrese un número válido.")

        # collect categorical feature inputs
        for var in self.categorical_vars:
            value = input(f"{var}: ")
            input_data[var] = value

        # build dataframe for the entered data
        input_df = pd.DataFrame([input_data])

        # separe numeric and categorical
        X_num = input_df[self.numeric_predictor_vars]
        X_cat = input_df[self.categorical_vars]

        # convert categorical columns to 'category' dtype
        for cat_col in self.categorical_vars:
            X_cat[cat_col] = X_cat[cat_col].astype("category")

        # conbine numeric and categorical features
        X_final = pd.concat([X_num, X_cat], axis=1)

        # run predictions
        prediction = self.model.predict(X_final)

        self.logger.info(
            f"\n Predicción de consumo de combustible: {prediction[0]:.2f} litros"
        )
        return prediction[0]


if __name__ == "__main__":
    # Variables predictoras
    numeric_predictor_vars = [
        "AvgSlopePercent",
        "TotalMeasuredTonnage",
        "Distance",
        "CycleDurationSeconds",
        "StageSequence",
        "TimeEfficiencyPercentage",
    ]

    # ✨ Todas las categóricas en una sola lista - XGBoost decide el mejor encoding
    categorical_vars = ["Destination", "DestinationType"]

    # Crear modelo con soporte nativo
    model = XGBoostModel(
        numeric_predictor_vars=numeric_predictor_vars,
        categorical_vars=categorical_vars,
        max_cat_to_onehot=4,  # <= 4 categorías: one-hot, > 4: optimal partitioning
    )

    try:
        # Pipeline de entrenamiento
        print("🚀 Iniciando entrenamiento con XGBoost Native Categorical Support")

        model.load_data()
        model.transform_cycles_data()
        resultados = model.train()

        # almacenar el modelo en un csv
        cycles_with_predictions = model.get_predictions()

        # Para guardar en CSV
        cycles_with_predictions.write_csv("xgboost_predictions.csv")

        model.print_feature_importance()
        # model.explain_model(plots=True)
        # model.predict_manual()

        model.plot_predictions("xgboost_predictions.png", resultados)
        # model.save_model()

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback

        traceback.print_exc()
