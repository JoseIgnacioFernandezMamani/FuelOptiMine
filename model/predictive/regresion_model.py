import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class FuelCycleRegressionAnalyzer:
    def __init__(self, csv_file="unified_data_T-210.csv") -> None:
        """
        Inicializa el analizador de ciclos de combustible
        """
        self.csv_file = csv_file
        self.data = pl.DataFrame()
        self.cycles = {}
        self.cycle_regressions = {}
        self.cycle_metrics = {}
        self.summary_stats = {}

        # Variables independientes para la regresión
        self.independent_vars = [
            "DistanceTraveled",
            "MeasuredTonnage",
            "SpeedAvg",
            "RPM",
            "SlopePercent",
            "Acceleration",
            "Distance",
        ]

        # Variable dependiente
        self.dependent_var = "FuelLevelLiters"

        # Variables que indican inicio de nuevo ciclo
        self.cycle_indicators = ["ValidFuel", "DeltaFuel", "BeforeAvg", "AfterAvg"]

    def load_data(self):
        """
        Carga los datos del archivo CSV
        """
        print(f"Cargando datos desde {self.csv_file}...")
        try:
            self.data = pl.read_csv(self.csv_file, try_parse_dates=True)
            print(f"Datos cargados exitosamente: {len(self.data)} registros")
            print(f"Columnas disponibles: {list(self.data.columns)}")
            return True
        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {self.csv_file}")
            return False
        except Exception as e:
            print(f"Error al cargar datos: {str(e)}")
            return False

    def identify_fuel_cycles(self):
        """
        Identifica los ciclos de combustible basado en las variables indicadoras
        """
        print("\nIdentificando ciclos de combustible...")

        # Convertir TimeStamp a datetime si existe
        if "TimeStamp" in self.data.columns:
            self.data["TimeStamp"] = pl.to_datetime(
                self.data["TimeStamp"], errors="coerce"
            )
            self.data = self.data.sort_values("TimeStamp").reset_index(drop=True)

        # Identificar inicios de ciclo (cuando todas las variables indicadoras no son null)
        cycle_starts = self.data[
            self.data[self.cycle_indicators].notna().all(axis=1)
        ].index.tolist()

        print(f"Encontrados {len(cycle_starts)} inicios de ciclo potenciales")

        # Crear diccionario de ciclos
        cycle_id = 0
        for i, start_idx in enumerate(cycle_starts):
            # Determinar fin del ciclo (siguiente inicio o final del dataset)
            if i < len(cycle_starts) - 1:
                end_idx = cycle_starts[i + 1] - 1
            else:
                end_idx = len(self.data) - 1

            # Extraer datos del ciclo
            cycle_data = self.data.iloc[start_idx : end_idx + 1].copy()

            # Filtrar solo registros válidos con FuelLevelLiters
            cycle_data = cycle_data.dropna(subset=[self.dependent_var])

            # Verificar que el ciclo tenga suficientes datos (mínimo 5 registros)
            if len(cycle_data) >= 5:
                # Calcular duración del ciclo
                if "TimeStamp" in cycle_data.columns:
                    duration = (
                        cycle_data["TimeStamp"].max() - cycle_data["TimeStamp"].min()
                    ).total_seconds() / 3600  # horas
                else:
                    duration = (
                        len(cycle_data) * 0.5 / 60
                    )  # asumiendo 30 segundos por registro, convertir a horas

                # Almacenar información del ciclo
                self.cycles[cycle_id] = {
                    "data": cycle_data,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "duration_hours": duration,
                    "initial_fuel": cycle_data[self.dependent_var].iloc[0],
                    "final_fuel": cycle_data[self.dependent_var].iloc[-1],
                    "fuel_consumed": cycle_data[self.dependent_var].iloc[0]
                    - cycle_data[self.dependent_var].iloc[-1],
                    "records_count": len(cycle_data),
                }
                cycle_id += 1

        print(f"Ciclos válidos identificados: {len(self.cycles)}")
        return len(self.cycles) > 0

    def perform_cycle_regressions(self):
        """
        Realiza regresión lineal multivariable para cada ciclo
        """
        print("\nRealizando regresiones lineales por ciclo...")

        successful_regressions = 0

        for cycle_id, cycle_info in self.cycles.items():
            cycle_data = cycle_info["data"]

            # Preparar datos para regresión
            # Variables independientes
            X = cycle_data[self.independent_vars].copy()
            # Variable dependiente
            y = cycle_data[self.dependent_var].copy()

            # Limpiar datos faltantes
            mask = X.notna().all(axis=1) & y.notna()
            X_clean = X[mask]
            y_clean = y[mask]

            # Verificar que tengamos suficientes datos
            if len(X_clean) < 3:
                print(f"Ciclo {cycle_id}: Datos insuficientes para regresión")
                continue

            try:
                # Crear y entrenar modelo de regresión
                model = LinearRegression()
                model.fit(X_clean, y_clean)

                # Realizar predicciones
                y_pred = model.predict(X_clean)

                # Calcular métricas
                mse = mean_squared_error(y_clean, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_clean, y_pred)
                r2 = r2_score(y_clean, y_pred)

                # Almacenar resultados
                self.cycle_regressions[cycle_id] = {
                    "model": model,
                    "X_data": X_clean,
                    "y_actual": y_clean,
                    "y_predicted": y_pred,
                    "coefficients": dict(zip(self.independent_vars, model.coef_)),
                    "intercept": model.intercept_,
                }

                self.cycle_metrics[cycle_id] = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2_score": r2,
                    "data_points": len(X_clean),
                    "duration_hours": cycle_info["duration_hours"],
                    "fuel_consumed": cycle_info["fuel_consumed"],
                    "consumption_rate": cycle_info["fuel_consumed"]
                    / max(cycle_info["duration_hours"], 0.1),
                }

                successful_regressions += 1

            except Exception as e:
                print(f"Error en regresión del ciclo {cycle_id}: {str(e)}")
                continue

        print(
            f"Regresiones completadas exitosamente: {successful_regressions}/{len(self.cycles)}"
        )
        return successful_regressions > 0

    def analyze_results(self):
        """
        Analiza los resultados de todas las regresiones
        """
        print("\nAnalizando resultados de regresiones...")

        if not self.cycle_metrics:
            print("No hay datos de métricas para analizar")
            return

        # Crear DataFrame con métricas de todos los ciclos
        metrics_df = pl.DataFrame(self.cycle_metrics).T

        # Estadísticas generales
        self.summary_stats = {
            "total_cycles": len(self.cycle_metrics),
            "avg_r2_score": metrics_df["r2_score"].mean(),
            "avg_rmse": metrics_df["rmse"].mean(),
            "avg_duration": metrics_df["duration_hours"].mean(),
            "avg_fuel_consumed": metrics_df["fuel_consumed"].mean(),
            "avg_consumption_rate": metrics_df["consumption_rate"].mean(),
        }

        # Análisis de coeficientes promedio
        all_coefficients = {}
        for cycle_id in self.cycle_regressions:
            coeffs = self.cycle_regressions[cycle_id]["coefficients"]
            for var, coeff in coeffs.items():
                if var not in all_coefficients:
                    all_coefficients[var] = []
                all_coefficients[var].append(coeff)

        # Coeficientes promedio
        avg_coefficients = {
            var: np.mean(coeffs) for var, coeffs in all_coefficients.items()
        }
        self.summary_stats["avg_coefficients"] = avg_coefficients

        # Mostrar resultados
        print("\n" + "=" * 50)
        print("RESUMEN DE ANÁLISIS DE CICLOS DE COMBUSTIBLE")
        print("=" * 50)
        print(f"Total de ciclos analizados: {self.summary_stats['total_cycles']}")
        print(f"R² promedio: {self.summary_stats['avg_r2_score']:.4f}")
        print(f"RMSE promedio: {self.summary_stats['avg_rmse']:.2f} litros")
        print(
            f"Duración promedio de ciclo: {self.summary_stats['avg_duration']:.2f} horas"
        )
        print(
            f"Combustible consumido promedio: {self.summary_stats['avg_fuel_consumed']:.2f} litros"
        )
        print(
            f"Tasa de consumo promedio: {self.summary_stats['avg_consumption_rate']:.2f} litros/hora"
        )

        print("\nCoeficientes promedio de regresión:")
        print("-" * 40)
        for var, coeff in avg_coefficients.items():
            print(f"{var:15}: {coeff:8.4f}")

        return metrics_df

    def predict_fuel_consumption(
        self,
        distance_traveled=0,
        measured_tonnage=0,
        speed_avg=0,
        rpm=0,
        slope_percent=0,
        acceleration=0,
        distance=0,
        duration_hours=1,
    ):
        """
        Predice el consumo de combustible basado en los modelos entrenados
        """
        if not self.summary_stats:
            print("Error: No hay modelos entrenados disponibles")
            return None

        # Usar coeficientes promedio para predicción
        coeffs = self.summary_stats["avg_coefficients"]

        # Calcular predicción usando regresión lineal promedio
        input_values = {
            "DistanceTraveled": distance_traveled,
            "MeasuredTonnage": measured_tonnage,
            "SpeedAvg": speed_avg,
            "RPM": rpm,
            "SlopePercent": slope_percent,
            "Acceleration": acceleration,
            "Distance": distance,
        }

        # Calcular consumo estimado
        estimated_consumption = 0
        for var, value in input_values.items():
            if var in coeffs:
                estimated_consumption += coeffs[var] * value

        # Ajustar por duración
        consumption_per_hour = estimated_consumption / max(duration_hours, 0.1)

        prediction_result = {
            "estimated_fuel_consumption_liters": abs(estimated_consumption),
            "consumption_rate_liters_per_hour": abs(consumption_per_hour),
            "input_parameters": input_values,
            "duration_hours": duration_hours,
        }

        return prediction_result

    def plot_cycle_analysis(self):
        """
        Genera gráficos de análisis de ciclos
        """
        if not self.cycle_metrics:
            print("No hay datos para graficar")
            return

        metrics_df = pl.DataFrame(self.cycle_metrics).T

        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Gráfico 1: R² Score por ciclo
        axes[0, 0].bar(range(len(metrics_df)), metrics_df["r2_score"])
        axes[0, 0].set_title("R² Score por Ciclo")
        axes[0, 0].set_xlabel("Ciclo ID")
        axes[0, 0].set_ylabel("R² Score")

        # Gráfico 2: Consumo vs Duración
        axes[0, 1].scatter(metrics_df["duration_hours"], metrics_df["fuel_consumed"])
        axes[0, 1].set_title("Consumo vs Duración del Ciclo")
        axes[0, 1].set_xlabel("Duración (horas)")
        axes[0, 1].set_ylabel("Combustible Consumido (litros)")

        # Gráfico 3: Tasa de Consumo
        axes[1, 0].hist(metrics_df["consumption_rate"], bins=10, alpha=0.7)
        axes[1, 0].set_title("Distribución de Tasa de Consumo")
        axes[1, 0].set_xlabel("Litros por Hora")
        axes[1, 0].set_ylabel("Frecuencia")

        # Gráfico 4: RMSE por ciclo
        axes[1, 1].plot(range(len(metrics_df)), metrics_df["rmse"], "o-")
        axes[1, 1].set_title("Error RMSE por Ciclo")
        axes[1, 1].set_xlabel("Ciclo ID")
        axes[1, 1].set_ylabel("RMSE (litros)")

        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self):
        """
        Ejecuta el análisis completo
        """
        print("INICIANDO ANÁLISIS COMPLETO DE CICLOS DE COMBUSTIBLE")
        print("=" * 60)

        # 1. Cargar datos
        if not self.load_data():
            return False

        # 2. Identificar ciclos
        if not self.identify_fuel_cycles():
            print("Error: No se pudieron identificar ciclos válidos")
            return False

        # 3. Realizar regresiones
        if not self.perform_cycle_regressions():
            print("Error: No se pudieron completar las regresiones")
            return False

        # 4. Analizar resultados
        metrics_df = self.analyze_results()

        # 5. Generar gráficos
        self.plot_cycle_analysis()

        return True


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del analizador
    analyzer = FuelCycleRegressionAnalyzer()

    # Ejecutar análisis completo
    success = analyzer.run_complete_analysis()

    if success:
        print("\n" + "=" * 60)
        print("EJEMPLO DE PREDICCIÓN")
        print("=" * 60)

        # Ejemplo de predicción
        prediction = analyzer.predict_fuel_consumption(
            distance_traveled=10.5,  # km
            measured_tonnage=150,  # toneladas
            speed_avg=25,  # km/h
            rpm=1800,  # revoluciones por minuto
            slope_percent=5.2,  # porcentaje de pendiente
            acceleration=0.5,  # m/s²
            distance=8.3,  # km
            duration_hours=2.0,  # horas
        )

        if prediction:
            print(
                f"Consumo estimado: {prediction['estimated_fuel_consumption_liters']:.2f} litros"
            )
            print(
                f"Tasa de consumo: {prediction['consumption_rate_liters_per_hour']:.2f} litros/hora"
            )
    else:
        print("El análisis no se pudo completar correctamente")
