import polars as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
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


class XGBoostNativeCategorical:
    """
    Clase para entrenamiento de un modelo XGBoost usando soporte nativo para variables categóricas.
    Más eficiente que el preprocesamiento manual (OneHot/Binary encoding).
    """

    def __init__(
        self,
        numeric_predictor_vars: List[str],
        categorical_vars: List[str],
        max_cat_to_onehot: int = 4,  # XGBoost decidirá automáticamente entre one-hot y partitioning
    ):
        """
        Inicializa la clase con soporte nativo para categóricas

        Args:
            numeric_predictor_vars: Lista de variables numéricas
            categorical_vars: Lista de variables categóricas
            max_cat_to_onehot: Umbral para decidir entre one-hot vs partitioning
                              (<=4 usa one-hot, >4 usa optimal partitioning)
        """
        self.categorical_vars = categorical_vars
        self.numeric_predictor_vars = numeric_predictor_vars
        self.max_cat_to_onehot = max_cat_to_onehot
        self.df = pl.DataFrame()
        self.cycles_data = pl.DataFrame()

        # Solo necesitamos escalador para numéricas y encoder ordinal para categóricas
        self.feature_scaler = StandardScaler()
        self.categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

        # Modelo XGBoost con soporte nativo para categóricas
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.08,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",  # Requerido para categorical support
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

        # Resultados y datos de test
        self.results = {}
        self.y_test = None
        self.y_pred = None
        self.feature_names = []

        # outliers
        self.outlier_mask = None
        # indidces
        self.test_indices_original = None

    def load_data(self, csv_path: str = "unified_data_T-210.csv"):
        """
        Cargar datos desde archivo CSV
        """
        print(f"Cargando datos desde {csv_path}...")
        df = pl.read_csv(csv_path, try_parse_dates=True)
        self.df = df.sort("SortTimestamp")
        return self.df

    def transform_cycles_data(self):
        """
        Procesar datos para identificar ciclos y calcular métricas de consumo de combustible
        """
        df = self.df.with_columns(
            [
                # marcar ciclos
                pl.when((pl.col("StageSequence") == 4) | (pl.col("StageSequence") == 8))
                .then(True)
                .otherwise(False)
                .alias("cycle_end"),
                # rolling median fuel level
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=10, min_samples=3, center=True)
                .alias("MedianFuelLevelLiters"),
                # destino consolidado
                pl.coalesce(["LoadingZone", "Destination"]).alias("Destination"),
            ]
        )

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

        result = result.with_columns(
            [
                # Mejorar el cálculo del consumo de combustible
                pl.when((pl.col("StartCycle") - pl.col("EndCycle")).abs() <= 500)
                .then((pl.col("StartCycle") - pl.col("EndCycle")).abs())
                .when(
                    (pl.col("StartCycle") - pl.col("EndCycle")) < 0
                )  # Si el tanque se llenó durante el ciclo
                .then(1)  # Valor mínimo
                .otherwise(10)  # Valor por defecto
                .alias("FuelConsumed"),
                (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                .dt.total_seconds()
                .abs()
                .alias("CycleDurationSeconds"),
            ]
        )

        # filtros básicos mejorados
        result = result.filter(
            (pl.col("Destination").str.strip_chars().str.len_chars() > 2)
            & (pl.col("Distance") > 0)
            & (pl.col("CycleDurationSeconds") > 120)
            & (pl.col("CycleDurationSeconds") < 21600)
            & (pl.col("FuelConsumed") >= 0.1)
            & (pl.col("FuelConsumed") <= 200)
        )

        # limpiar columnas
        cols_to_clean = self.numeric_predictor_vars + ["FuelConsumed"]
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
        print(f"Datos procesados: {len(result)} ciclos válidos")

    def prepare_data(self):
        """
        Prepara datos usando soporte nativo de XGBoost para categóricas
        """
        # Convertir a pandas
        df = self.cycles_data.to_pandas()

        # Separar variables numéricas y categóricas
        X_numeric = df[self.numeric_predictor_vars]
        X_categorical = df[self.categorical_vars]
        y = df["FuelConsumed"]

        # ✨ PASO CLAVE: Convertir categóricas al tipo 'category' de pandas
        # Esto es lo que XGBoost necesita para reconocer las variables categóricas
        for cat_col in self.categorical_vars:
            X_categorical[cat_col] = X_categorical[cat_col].astype("category")
            print(
                f"Variable '{cat_col}' convertida a categoría con {X_categorical[cat_col].nunique()} valores únicos"
            )

        # Escalar solo las variables numéricas
        X_numeric_scaled = self.feature_scaler.fit_transform(X_numeric)
        X_numeric_df = pd.DataFrame(
            X_numeric_scaled, columns=self.numeric_predictor_vars, index=df.index
        )

        # ✨ Combinar numéricas escaladas con categóricas (manteniendo el tipo category)
        X_final = pd.concat([X_numeric_df, X_categorical], axis=1)

        # Eliminar outliers usando solo las variables numéricas para detección
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_mask = iso_forest.fit_predict(X_numeric_scaled) == 1
        X_clean = X_final[outlier_mask]
        y_clean = y[outlier_mask]

        # almacenar mascara para outliers
        self.outlier_mask = outlier_mask

        # Guardar nombres de features
        self.feature_names = list(X_final.columns)

        print(f"Datos después de limpieza: {len(X_clean)} registros")
        print(f"Features numéricas: {self.numeric_predictor_vars}")
        print(f"Features categóricas: {self.categorical_vars}")
        print(f"Dtypes finales:")
        for col in X_final.columns:
            print(f"  {col}: {X_final[col].dtype}")

        return train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    def train(self):
        """
        Entrenar modelo XGBoost con soporte nativo para categóricas
        """
        X_train, X_test, y_train, y_test = self.prepare_data()

        # agregar indies originales de test
        self.test_indices_original = X_test.index

        print(f"Entrenando con {len(X_train)} registros...")
        print(f"Dimensiones: {X_train.shape}")

        # ✨ XGBoost automáticamente detecta las variables categóricas por el dtype 'category'
        # y decide usar one-hot encoding o optimal partitioning según max_cat_to_onehot
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50,
        )

        print(f"Mejor iteración: {self.model.best_iteration}")

        # Mostrar información sobre cómo XGBoost procesó las categóricas
        print(
            f"Tipos de features detectados por XGBoost: {self.model.get_booster().feature_types}"
        )

        y_pred = self.model.predict(X_test)

        # Cálculo de métricas
        y_pred_non_negative = np.maximum(y_pred, 0.1)
        y_test_non_negative = np.maximum(y_test, 0.1)

        try:
            rmsle = np.sqrt(
                mean_squared_log_error(y_test_non_negative, y_pred_non_negative)
            )
        except ValueError:
            rmsle = float("inf")

        self.y_test = y_test
        self.y_pred = y_pred

        mape_safe = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.1))) * 100

        self.results = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAPE_Safe": mape_safe,
            "MedianAE": median_absolute_error(y_test, y_pred),
            "RMSLE": rmsle,
            "ExplainedVar": explained_variance_score(y_test, y_pred),
        }

        return self.results

    def plot_predictions(self, save_path: str = None):
        """
        Crear gráfica de predicciones vs valores reales
        """
        if self.y_test is None or self.y_pred is None:
            print("Error: Debe entrenar el modelo primero")
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
        Obtener importancia de características
        """
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def print_feature_importance(self):
        """
        Imprimir importancia de características con información adicional
        """
        importance_df = self.get_feature_importance()
        print("\n===== Importancia de Características (XGBoost Nativo) =====")

        # Separar por tipo de variable
        numeric_features = importance_df[
            importance_df["feature"].isin(self.numeric_predictor_vars)
        ]
        categorical_features = importance_df[
            importance_df["feature"].isin(self.categorical_vars)
        ]

        print("\n📊 Variables Numéricas:")
        for _, row in numeric_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        print("\n🏷️ Variables Categóricas (Procesamiento Nativo):")
        for _, row in categorical_features.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    def save_model(self, path: str = "xgboost_native_categorical.json"):
        """
        Guardar modelo (debe ser JSON para preservar info categórica)
        """
        self.model.get_booster().save_model(path)
        print(f"✅ Modelo guardado en: {path}")
        print("💡 IMPORTANTE: Usar formato JSON para preservar información categórica")

    def get_cycles_with_predictions(self) -> pl.DataFrame:
        """
        Genera un DataFrame con los datos de ciclo completos incluyendo predicciones y residuos

        Returns:
            pl.DataFrame: DataFrame con todas las columnas solicitadas incluyendo predicciones
        """
        if not hasattr(self, "test_indices_original") or not hasattr(
            self, "outlier_mask"
        ):
            raise Exception("Debe entrenar el modelo primero usando el método train()")

        # Crear DataFrame base con todos los ciclos
        result = self.cycles_data.clone()

        # Añadir columnas de outliers/inliers
        result = result.with_columns(
            [
                pl.lit(self.outlier_mask).alias("is_inlier"),
                pl.lit(~self.outlier_mask).alias("is_outlier"),
            ]
        )

        # Inicializar arrays para predicciones y residuos
        all_predictions = np.full(len(result), np.nan)
        all_residuals = np.full(len(result), np.nan)

        # Llenar con valores de test set donde corresponda
        all_predictions[self.test_indices_original] = self.y_pred
        all_residuals[self.test_indices_original] = self.y_test - self.y_pred

        # Añadir predicciones y residuos al DataFrame
        result = result.with_columns(
            [
                pl.Series("predicted_fuel_xgboost", all_predictions),
                pl.Series("residual", all_residuals),
            ]
        )

        # Seleccionar y ordenar columnas según lo solicitado
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
                "predicted_fuel_xgboost",
                "residual",
            ]
        )


if __name__ == "__main__":
    # Variables predictoras
    numeric_predictor_vars = [
        "AverageSpeed",
        "AvgSlopePercent",
        "AvgAcceleration",
        "TotalMeasuredTonnage",
        "Distance",
        "CycleDurationSeconds",
    ]

    # ✨ Todas las categóricas en una sola lista - XGBoost decide el mejor encoding
    categorical_vars = ["Destination", "StageSequence"]

    # Crear modelo con soporte nativo
    model = XGBoostNativeCategorical(
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
        cycles_with_predictions = model.get_cycles_with_predictions()
        print(cycles_with_predictions.head())

        # Para guardar en CSV
        cycles_with_predictions.write_csv("ciclos_con_predicciones.csv")

        # Resultados
        print("\n===== Resultados XGBoost Native Categorical =====")
        for k, v in resultados.items():
            print(f"{k}: {v:.4f}")

        model.print_feature_importance()
        model.plot_predictions("xgboost_native_predictions.png")
        model.save_model()

        print("\n✨ Ventajas del enfoque nativo:")
        print("  • No necesita preprocesamiento manual de categóricas")
        print("  • XGBoost decide automáticamente entre one-hot y partitioning")
        print("  • Mejor manejo de valores desconocidos")
        print("  • Splits más eficientes para categóricas con muchos valores")
        print("  • Mayor interpretabilidad del modelo")

    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback

        traceback.print_exc()
