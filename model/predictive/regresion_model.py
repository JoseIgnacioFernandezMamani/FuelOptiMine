import torch
import torch.nn as nn
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuelLinearRegressionModel:
    """
    Clase para regresión lineal de combustible basada en delta_fuel y FuelLevelLiters.
    Implementa tanto regresión lineal tradicional como RANSAC para manejar outliers.
    """

    def __init__(self, use_ransac: bool = True, random_state: int = 42):
        """
        Inicializa el modelo de regresión lineal.

        Args:
            use_ransac: Si usar RANSAC para manejo robusto de outliers
            random_state: Semilla para reproducibilidad
        """
        self.use_ransac = use_ransac
        self.random_state = random_state
        self.lr_model = linear_model.LinearRegression()
        self.ransac_model = (
            linear_model.RANSACRegressor(random_state=random_state)
            if use_ransac
            else None
        )
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.is_trained = False

    def load_data(self, data_path: Path) -> pl.DataFrame:
        """
        Carga los datos desde el archivo CSV usando Polars.

        Args:
            data_path: Ruta al archivo CSV

        Returns:
            DataFrame con los datos cargados
        """
        try:
            logger.info(f"Cargando datos desde: {data_path}")
            self.data = pl.read_csv(data_path)
            logger.info(
                f"Datos cargados: {self.data.shape[0]} filas, {self.data.shape[1]} columnas"
            )
            return self.data
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise

    def preprocess_data(
        self, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocesa los datos para entrenamiento.

        Args:
            test_size: Proporción de datos para prueba

        Returns:
            Tupla con X_train, X_test, y_train, y_test
        """
        if self.data is None:
            raise ValueError("Primero debe cargar los datos con load_data()")

        # Filtrar datos válidos (no nulos)
        clean_data = self.data.filter(
            (pl.col("delta_fuel").is_not_null())
            & (pl.col("FuelLevelLiters").is_not_null())
            & (pl.col("delta_fuel") != 0)  # Evitar divisiones por cero
            & (pl.col("FuelLevelLiters") > 0)  # Evitar valores negativos de combustible
        )

        logger.info(f"Datos después de limpieza: {clean_data.shape[0]} filas")

        # Extraer características (X) y variable objetivo (y)
        X = clean_data.select("delta_fuel").to_numpy()
        y = clean_data.select("FuelLevelLiters").to_numpy().flatten()

        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        logger.info(f"Datos de entrenamiento: {self.X_train.shape[0]} muestras")
        logger.info(f"Datos de prueba: {self.X_test.shape[0]} muestras")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self) -> dict:
        """
        Entrena los modelos de regresión lineal.

        Returns:
            Diccionario con métricas de entrenamiento
        """
        if self.X_train is None:
            raise ValueError("Primero debe preprocesar los datos con preprocess_data()")

        logger.info("Iniciando entrenamiento...")

        # Entrenar regresión lineal tradicional
        self.lr_model.fit(self.X_train, self.y_train)

        # Entrenar RANSAC si está habilitado
        if self.use_ransac and self.ransac_model is not None:
            self.ransac_model.fit(self.X_train, self.y_train)

        self.is_trained = True
        logger.info("Entrenamiento completado")

        # Calcular métricas de entrenamiento
        metrics = self.evaluate()
        return metrics

    def predict(self, X: np.ndarray, use_ransac: bool = None) -> np.ndarray:
        """
        Realiza predicciones usando el modelo entrenado.

        Args:
            X: Datos de entrada (delta_fuel)
            use_ransac: Si usar RANSAC (por defecto usa la configuración de la clase)

        Returns:
            Predicciones de FuelLevelLiters
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        use_ransac = use_ransac if use_ransac is not None else self.use_ransac

        if use_ransac and self.ransac_model is not None:
            return self.ransac_model.predict(X)
        else:
            return self.lr_model.predict(X)

    def evaluate(self) -> dict:
        """
        Evalúa el modelo en el conjunto de prueba.

        Returns:
            Diccionario con métricas de evaluación
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        # Predicciones con regresión lineal
        y_pred_lr = self.lr_model.predict(self.X_test)
        mse_lr = mean_squared_error(self.y_test, y_pred_lr)
        r2_lr = r2_score(self.y_test, y_pred_lr)

        metrics = {
            "linear_regression": {
                "mse": mse_lr,
                "r2": r2_lr,
                "coef": self.lr_model.coef_[0],
                "intercept": self.lr_model.intercept_,
            }
        }

        # Métricas para RANSAC si está disponible
        if self.use_ransac and self.ransac_model is not None:
            y_pred_ransac = self.ransac_model.predict(self.X_test)
            mse_ransac = mean_squared_error(self.y_test, y_pred_ransac)
            r2_ransac = r2_score(self.y_test, y_pred_ransac)

            metrics["ransac"] = {
                "mse": mse_ransac,
                "r2": r2_ransac,
                "coef": self.ransac_model.estimator_.coef_[0],
                "intercept": self.ransac_model.estimator_.intercept_,
                "n_inliers": np.sum(self.ransac_model.inlier_mask_),
                "n_outliers": np.sum(~self.ransac_model.inlier_mask_),
            }

        return metrics

    def plot_results(self, save_path: Optional[Path] = None):
        """
        Visualiza los resultados de la regresión.

        Args:
            save_path: Ruta para guardar la gráfica (opcional)
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        # Combinar datos de entrenamiento y prueba para visualización completa
        X_all = np.vstack([self.X_train, self.X_test])
        y_all = np.hstack([self.y_train, self.y_test])

        # Crear rango para las líneas de predicción
        line_X = np.arange(X_all.min(), X_all.max()).reshape(-1, 1)

        # Predicciones
        line_y_lr = self.lr_model.predict(line_X)

        plt.figure(figsize=(12, 8))

        if self.use_ransac and self.ransac_model is not None:
            # Identificar inliers y outliers
            self.ransac_model.fit(
                X_all, y_all
            )  # Re-entrenar con todos los datos para visualización
            inlier_mask = self.ransac_model.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)

            line_y_ransac = self.ransac_model.predict(line_X)

            # Plot puntos
            plt.scatter(
                X_all[inlier_mask],
                y_all[inlier_mask],
                color="yellowgreen",
                marker=".",
                label="Inliers",
                alpha=0.6,
            )
            plt.scatter(
                X_all[outlier_mask],
                y_all[outlier_mask],
                color="gold",
                marker=".",
                label="Outliers",
                alpha=0.8,
            )

            # Plot líneas de regresión
            plt.plot(
                line_X, line_y_lr, color="navy", linewidth=2, label="Linear Regression"
            )
            plt.plot(
                line_X,
                line_y_ransac,
                color="cornflowerblue",
                linewidth=2,
                label="RANSAC Regression",
            )
        else:
            # Solo regresión lineal
            plt.scatter(X_all, y_all, color="blue", alpha=0.6, label="Data points")
            plt.plot(
                line_X, line_y_lr, color="red", linewidth=2, label="Linear Regression"
            )

        plt.xlabel("Delta Fuel (L)")
        plt.ylabel("Fuel Level (L)")
        plt.title("Regresión Lineal: Delta Fuel vs Nivel de Combustible")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Gráfica guardada en: {save_path}")

        plt.show()

    def get_model_summary(self) -> dict:
        """
        Obtiene un resumen del modelo entrenado.

        Returns:
            Diccionario con información del modelo
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")

        summary = {
            "data_shape": self.data.shape if self.data is not None else None,
            "training_samples": len(self.X_train),
            "test_samples": len(self.X_test),
            "use_ransac": self.use_ransac,
            "models_trained": ["linear_regression"],
        }

        if self.use_ransac:
            summary["models_trained"].append("ransac")

        return summary


# PyTorch Linear Regression Model (alternativa)
class PyTorchLinearRegressionModel(nn.Module):
    """
    Modelo de regresión lineal usando PyTorch como alternativa.
    """

    def __init__(self, input_size: int = 1):
        super(PyTorchLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar rutas
    current_file: Path = Path(__file__).resolve().parent.parent.parent
    data_path: Path = (
        current_file / "frontend/web/app/correlated_events/all_correlated_events.csv"
    )

    print(data_path)
    # Crear y usar el modelo
    fuel_regressor = FuelLinearRegressionModel(use_ransac=True, random_state=42)

    try:
        # Cargar y preprocesar datos
        data = fuel_regressor.load_data(data_path)
        fuel_regressor.preprocess_data(test_size=0.2)

        # Entrenar modelo
        metrics = fuel_regressor.train()

        # Mostrar resultados
        print("\n=== MÉTRICAS DE EVALUACIÓN ===")
        for model_name, model_metrics in metrics.items():
            print(f"\n{model_name.upper()}:")
            for metric_name, value in model_metrics.items():
                print(f"  {metric_name}: {value}")

        # Visualizar resultados
        fuel_regressor.plot_results()

        # Mostrar resumen del modelo
        summary = fuel_regressor.get_model_summary()
        print("\n=== RESUMEN DEL MODELO ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

        # Ejemplo de predicción
        new_delta_fuel = np.array([[50.0], [100.0], [150.0]])  # Ejemplos de delta_fuel
        predictions = fuel_regressor.predict(new_delta_fuel, use_ransac=True)

        print("\n=== PREDICCIONES DE EJEMPLO ===")
        for i, (delta, pred) in enumerate(zip(new_delta_fuel.flatten(), predictions)):
            print(f"Delta Fuel: {delta}L -> Nivel Predicho: {pred:.2f}L")

    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        raise
