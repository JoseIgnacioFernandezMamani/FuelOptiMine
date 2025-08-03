import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def load_and_process_data(filename):
    """
    Carga el archivo CSV y extrae las columnas necesarias
    """
    try:
        df = pd.read_csv(filename)
        print(f"Archivo cargado exitosamente. Forma: {df.shape}")
        print(f"Columnas disponibles: {list(df.columns)}")

        # Verificar que las columnas necesarias existen
        required_columns = ["delta_fuel", "FuelLevelLiters"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"ERROR: Columnas faltantes: {missing_columns}")
            return None

        # Extraer las columnas necesarias y eliminar valores nulos
        data = df[required_columns].dropna()
        print(f"Datos después de eliminar valores nulos: {data.shape}")

        return data

    except FileNotFoundError:
        print(f"ERROR: No se encontró el archivo {filename}")
        return None
    except Exception as e:
        print(f"ERROR al cargar el archivo: {e}")
        return None


def calculate_fuel_discrepancy(delta_fuel, fuel_level_liters, adjustment_value):
    """
    Calcula la discrepancia de combustible dado un valor de ajuste

    fuel_discrepancy = |delta_fuel + adjustment_value - FuelLevelLiters|
    """
    adjusted_delta_fuel = delta_fuel + adjustment_value
    discrepancy = np.abs(adjusted_delta_fuel - fuel_level_liters)
    return discrepancy


def objective_function(adjustment_value, delta_fuel, fuel_level_liters):
    """
    Función objetivo: promedio de la discrepancia de combustible
    """
    discrepancy = calculate_fuel_discrepancy(
        delta_fuel, fuel_level_liters, adjustment_value
    )
    return np.mean(discrepancy)


def find_optimal_adjustment(data):
    """
    Encuentra el valor óptimo de ajuste usando optimización
    """
    delta_fuel = data["delta_fuel"].values
    fuel_level_liters = data["FuelLevelLiters"].values

    # Definir el rango de búsqueda basado en los datos
    data_range = np.max(fuel_level_liters) - np.min(delta_fuel)
    search_bounds = (-data_range, data_range)

    print(f"Buscando valor óptimo en el rango: {search_bounds}")

    # Optimización usando minimize_scalar
    result = minimize_scalar(
        objective_function,
        bounds=search_bounds,
        method="bounded",
        args=(delta_fuel, fuel_level_liters),
    )

    return result


def analyze_results(data, optimal_adjustment):
    """
    Analiza los resultados de la optimización
    """
    delta_fuel = data["delta_fuel"].values
    fuel_level_liters = data["FuelLevelLiters"].values

    # Calcular discrepancia original (sin ajuste)
    original_discrepancy = calculate_fuel_discrepancy(delta_fuel, fuel_level_liters, 0)
    original_avg_discrepancy = np.mean(original_discrepancy)

    # Calcular discrepancia optimizada
    optimized_discrepancy = calculate_fuel_discrepancy(
        delta_fuel, fuel_level_liters, optimal_adjustment
    )
    optimized_avg_discrepancy = np.mean(optimized_discrepancy)

    print("\n" + "=" * 50)
    print("RESULTADOS DE LA OPTIMIZACIÓN")
    print("=" * 50)
    print(f"Valor constante óptimo a sumar: {optimal_adjustment:.6f}")
    print(f"Discrepancia promedio original: {original_avg_discrepancy:.6f}")
    print(f"Discrepancia promedio optimizada: {optimized_avg_discrepancy:.6f}")
    print(
        f"Mejora obtenida: {original_avg_discrepancy - optimized_avg_discrepancy:.6f}"
    )
    print(
        f"Reducción porcentual: {((original_avg_discrepancy - optimized_avg_discrepancy) / original_avg_discrepancy * 100):.2f}%"
    )

    return {
        "optimal_adjustment": optimal_adjustment,
        "original_avg_discrepancy": original_avg_discrepancy,
        "optimized_avg_discrepancy": optimized_avg_discrepancy,
        "original_discrepancy": original_discrepancy,
        "optimized_discrepancy": optimized_discrepancy,
    }

def plot_results(data, results):
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Gráfico 1: en la posición (0,0)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(
        results["original_discrepancy"],
        bins=50,
        alpha=0.7,
        label="Original",
        color="red",
    )
    ax1.hist(
        results["optimized_discrepancy"],
        bins=50,
        alpha=0.7,
        label="Optimizada",
        color="green",
    )
    ax1.set_xlabel("Discrepancia de Combustible")
    ax1.set_ylabel("Frecuencia")
    ax1.set_title("Distribución de Discrepancias")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: en la posición (0,1)
    ax2 = fig.add_subplot(gs[0, 1])
    adjustment_range = np.linspace(-50, 50, 200)
    objective_values = [
        objective_function(
            adj, data["delta_fuel"].values, data["FuelLevelLiters"].values
        )
        for adj in adjustment_range
    ]
    ax2.plot(adjustment_range, objective_values, "b-", linewidth=2)
    ax2.axvline(
        results["optimal_adjustment"],
        color="red",
        linestyle="--",
        label=f'Óptimo: {results["optimal_adjustment"]:.3f}',
    )
    ax2.set_xlabel("Valor de Ajuste")
    ax2.set_ylabel("Discrepancia Promedio")
    ax2.set_title("Función Objetivo")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Gráfico 3: ocupa toda la fila inferior (fila 1, columnas 0 y 1)
    ax3 = fig.add_subplot(gs[1, :])
    adjusted_delta_fuel = data["delta_fuel"] + results["optimal_adjustment"]
    ax3.scatter(adjusted_delta_fuel, data["FuelLevelLiters"], alpha=0.6, s=10)
    ax3.plot(
        [adjusted_delta_fuel.min(), adjusted_delta_fuel.max()],
        [adjusted_delta_fuel.min(), adjusted_delta_fuel.max()],
        "r--",
        label="Línea perfecta",
    )
    ax3.set_xlabel("Delta Fuel Ajustado")
    ax3.set_ylabel("Fuel Level Liters")
    ax3.set_title("Datos Ajustados")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """
    Función principal que ejecuta todo el análisis
    """
    filename = "all_correlated_events.csv"

    # Cargar datos
    data = load_and_process_data(filename)
    if data is None:
        return

    print(f"\nEstadísticas básicas:")
    print(
        f"Delta Fuel - Media: {data['delta_fuel'].mean():.3f}, Std: {data['delta_fuel'].std():.3f}"
    )
    print(
        f"Fuel Level Liters - Media: {data['FuelLevelLiters'].mean():.3f}, Std: {data['FuelLevelLiters'].std():.3f}"
    )

    # Encontrar el valor óptimo
    optimization_result = find_optimal_adjustment(data)

    if optimization_result.success:
        optimal_adjustment = optimization_result.x

        # Analizar resultados
        results = analyze_results(data, optimal_adjustment)

        # Crear gráficos
        plot_results(data, results)

        # Guardar resultados
        output_data = data.copy()
        output_data["delta_fuel_adjusted"] = data["delta_fuel"] + optimal_adjustment
        output_data["fuel_discrepancy_original"] = results["original_discrepancy"]
        output_data["fuel_discrepancy_optimized"] = results["optimized_discrepancy"]

        output_filename = "fuel_optimization_results.csv"
        output_data.to_csv(output_filename, index=False)
        print(f"\nResultados guardados en: {output_filename}")

    else:
        print("ERROR: La optimización no convergió")
        print(f"Mensaje: {optimization_result.message}")


if __name__ == "__main__":
    main()
