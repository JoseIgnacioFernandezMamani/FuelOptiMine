from PIL.Image import logger
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("xgb.log", mode="a"),  # guarda en archivo
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
            n_estimators=1000,
            learning_rate=0.08,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="gpu_hist",  # Requerido para categorical support
            device="cuda",
            random_state=42,
            eval_metric="mae",
            early_stopping_rounds=100,
            reg_alpha=0.1,
            reg_lambda=1,
            verbosity=1,
            # ✨ Parámetros clave para soporte nativo de categóricas
            enable_categorical=True,
            max_cat_to_onehot=max_cat_to_onehot,  # Control automático de encoding
        )

        # Results and testing data
        self.results = {}
        self.y_test = None
        self.y_pred = None
        self.feature_names = []

        # outliers detection model
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)

        # store test data
        self.X_test = None

    def load_data(self, csv_path: str = "unified_data_T-210.csv"):
        """
        hardcoding for moment, should be improved
        """
        model = LinearRegressionModel()
        results = model.train_models()
        self.df = model.get_predictions()

    def transform_cycles_data(self):
        """
        Process data to identify cycles and calculate fuel consumption metrics.
        """
        logger.info("Transformando datos de cycle")

        df = self.df
        result = df.with_columns(
            [
                # Calculation of fuel consumption, with default values
                pl.when(
                    ((pl.col("StartCycle") - pl.col("EndCycle")).abs() <= 500)
                    & ((pl.col("StartCycle") - pl.col("EndCycle")).abs() >= 5)
                )
                .then((pl.col("StartCycle") - pl.col("EndCycle")).abs())
                .when(((pl.col("StartCycle") - pl.col("EndCycle")) < 5))
                .then(
                    pl.col("PredictedFuelConsumption")
                )  # Default value calculated from the regression model
                .otherwise(5)  # default minimum consumption
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
        logger.info(f"Datos procesados: {len(result)} ciclos válidos.")

    def prepare_data(self):
        """
        Prepare data for XGBoost training using native categorical support.
        """
        logger.info("Preparing data for training...")

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
        X_final = pd.concat([X_numeric, X_categorical], axis=1)

        # outlier detection using numeric features
        self.iso_forest.fit(X_numeric)
        outlier_mask = self.iso_forest.predict(X_numeric) == 1
        X_clean = X_final[outlier_mask]
        y_clean = y[outlier_mask]

        # save feature names
        self.feature_names = list(X_final.columns)

        logger.info(f"Data after cleaning: {len(X_clean)} records.")
        logger.info(f"Numeric features: {self.numeric_predictor_vars}")
        logger.info(f"Categorical features: {self.categorical_vars}")
        logger.info("Final dtypes:")
        for col in X_final.columns:
            logger.info(f"  {col}: {X_final[col].dtype}")

        return train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

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

        logger.info(f"Entrenando con {len(X_train)} records...")

        # train model and evaluate simultaneously
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50,
        )

        logger.info(f"Mejor iteración: {self.model.best_iteration}")

        # Mostrar información sobre cómo XGBoost procesó las categóricas
        logger.info(
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

        # store evaluation metrics
        self.results = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAPE_Safe": mape_safe,
            "MedianAE": median_absolute_error(y_test, y_pred),
            "RMSLE": rmsle,
            "ExplainedVar": explained_variance_score(y_test, y_pred),
        }

        logger.info("Training completed. Evaluation metrics calculated successfully.")
        return self.results

    def plot_predictions(self, save_path: str = None):
        """
        Generate visual plots comparing model predictions against actual values.

        This method creates two visualizations:
        1. Time series plot showing the progression of predictions vs real values.
        2. Scatter plot comparing predictions against real values with a "perfect fit" line.

        Metrics (R² and RMSE) are displayed on the scatter plot for quick interpretation.

        Args:
            save_path (str, optional): File path to save the generated plots.
                                       If None, plots are only displayed.

        Logging:
            - Logs error if model is not trained before plotting.
            - Logs confirmation if plots are saved to disk.
        """
        if self.y_test is None or self.y_pred is None:
            logger.error("Model must be trained before plotting predictions.")
            return

        plt.style.use("default")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        indices = range(1, len(self.y_test) + 1)

        # Gráfica 1: Series de tiempo
        ax1.plot(
            indices,
            self.y_test,
            "o-",
            color="blue",
            alpha=0.7,
            label="Valores Reales",
            markersize=4,
            linewidth=1,
        )
        ax1.plot(
            indices,
            self.y_pred,
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
        ax2.scatter(self.y_test, self.y_pred, alpha=0.6, color="green")

        min_val = min(min(self.y_test), min(self.y_pred))
        max_val = max(max(self.y_test), max(self.y_pred))
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

        r2 = self.results["R2"]
        rmse = self.results["RMSE"]
        ax2.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}\nRMSE = {rmse:.2f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Gráfica guardada en: {save_path}")

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
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        logger.info("Feature importance extracted successfully.")

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
        logger.info("\n===== Importancia de Características (XGBoost Nativo) =====")

        # split by type
        numeric_features = importance_df[
            importance_df["feature"].isin(self.numeric_predictor_vars)
        ]
        categorical_features = importance_df[
            importance_df["feature"].isin(self.categorical_vars)
        ]

        logger.info("\n Variables Numéricas:")
        for _, row in numeric_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        logger.info("\n Variables Categóricas (Procesamiento Nativo):")
        for _, row in categorical_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

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
        logger.info(f" Modelo guardado en: {path}")
        logger.info(
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
        X_numeric = df_all[self.numeric_predictor_vars]
        X_categorical = df_all[self.categorical_vars]

        # Convert categorical variables to 'category' dtype (same as in training)
        for cat_col in self.categorical_vars:
            X_categorical[cat_col] = X_categorical[cat_col].astype("category")

        # Combine numeric and categorical features
        X_all = pd.concat([X_numeric, X_categorical], axis=1)

        # run predictions for all available data
        y_pred_all = self.model.predict(X_all)
        y_true_all = df_all["FuelConsumed"].values

        # compute residuals
        residuals_all = y_true_all - y_pred_all

        # apply outlier detection on numeric features
        outlier_predictions = self.iso_forest.predict(X_numeric)

        is_inlier = outlier_predictions == 1
        is_outlier = outlier_predictions == -1

        # convert back to polars and enrich dataset
        result = self.cycles_data.with_columns(
            [
                pl.Series("PredictedFuelXGBoost", y_pred_all),
                pl.Series("residual", residuals_all),
                pl.Series("is_inlier", is_inlier),
                pl.Series("is_outlier", is_outlier),
            ]
        )

        logger.info("Predictions, residuals, and outlier flags added to dataset.")
        return result.select(
            [
                "cycle_group",
                "TimeStampIni",
                "TimeStampFin",
                "ShiftDate",
                "Equipment",
                "TruckFleet",
                "StartCycle",
                "EndCycle",
                "AvgSpeed",
                "AvgSlopePercent",
                "AvgAcceleration",
                "AvgTimeEfficiencyPercentage",
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
        )

    def explain_model(self, max_display: int = 15):
        """
        Generate SHAP-based explanations for the trained XGBoost model.

        This method uses SHAP (SHapley Additive exPlanations) to interpret the trained
        XGBoost model. SHAP values provide insights into how much each feature contributes
        to the model’s predictions, both on average and per individual prediction.

        The method performs the following steps:
            1. Verifies that the model and test data exist.
            2. Constructs a SHAP TreeExplainer, tailored for tree-based models like XGBoost.
            3. Calculates SHAP values for the test dataset.
            4. Aggregates and ranks features based on their mean absolute contribution.
            5. Logs the top contributing features (limited by `max_display`).
            6. Evaluates the additivity property of SHAP explanations by comparing
               the reconstructed prediction (SHAP sum + expected value) against
               the model’s raw predictions.
            7. Logs the relative mean and maximum additivity errors as a measure
               of explanation accuracy.

        Args:
            max_display (int, optional): Maximum number of top features to display.
                                         Defaults to 15.

        Returns:
            None. The function logs detailed SHAP explanation results.
        """
        if not hasattr(self, "model") or self.model is None:
            logger.error("❌ Error: El modelo no ha sido entrenado todavía.")
            return

        if not hasattr(self, "X_test") or self.X_test is None:
            logger.error("❌ Error: No existen datos de test. Ejecute train() primero.")
            return

        logger.info("\n Calculando explicaciones con SHAP...")

        # Crear shap explainer
        explainer = shap.TreeExplainer(
            self.model, feature_perturbation="interventional"
        )

        # compute shap values for test data
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

        logger.info("\n===== Aporte de variables según SHAP =====")
        for _, row in importance_df.head(max_display).iterrows():
            logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

        # validate shap additivity property (prediction = sum(shap) + expected_value)
        shap_sum = shap_values.sum(axis=1) + explainer.expected_value
        preds = self.model.predict(self.X_test)
        additivity_error = np.abs(shap_sum - preds) / (np.abs(preds) + 1e-8)
        logger.info("Media error relativo de aditividad:", additivity_error.mean())
        logger.info("Máximo error relativo:", additivity_error.max())
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
            logger.error(
                "❌ Error: El modelo no está entrenado. Ejecute train() primero."
            )
            return

        logger.info("\n Ingrese los valores para la predicción:")

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

        logger.info(
            f"\n Predicción de consumo de combustible: {prediction[0]:.2f} litros"
        )
        return prediction[0]

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


if __name__ == "__main__":
    # Variables predictoras
    numeric_predictor_vars = [
        "AvgSpeed",
        "AvgSlopePercent",
        "AvgAcceleration",
        "TotalMeasuredTonnage",
        "Distance",
        "CycleDurationSeconds",
        "StageSequence",
    ]

    # ✨ Todas las categóricas en una sola lista - XGBoost decide el mejor encoding
    categorical_vars = ["Destination"]

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
        print(cycles_with_predictions.head())

        # Para guardar en CSV
        cycles_with_predictions.write_csv("xgboost_predictions.csv")

        # Resultados
        print("\n===== Resultados XGBoost Native Categorical =====")
        for k, v in resultados.items():
            print(f"{k}: {v:.4f}")

        # model.print_feature_importance()
        # model.explain_model()
        # model.predict_manual()

        # model.plot_predictions("xgboost_predictions.png")
        # model.save_model()

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback

        traceback.print_exc()
