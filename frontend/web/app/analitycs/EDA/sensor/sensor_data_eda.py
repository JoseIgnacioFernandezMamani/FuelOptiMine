from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import polars as pl
from pathlib import Path
from typing import Any, Dict
from analitycs.EDA.config.settings import DATA_DIR
import numpy as np
from datetime import datetime


class SensorDataEDA:
    def __init__(self) -> None:
        self.data_path = Path(DATA_DIR) / "T-211_sensor.csv"
        self.sensor_df: pl.DataFrame = None
        self._stats_cache: Dict[str, Dict[str, Any]] = None
        self._data_loaded: bool = False
        self._stats_generated: bool = False

    def get_dataframe(self) -> pl.DataFrame:
        """Get the loaded DataFrame"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")
        return self.sensor_df

    def _load_sensor_data(self):
        """Load and preprocess sensor data with proper type casting
        en un  futuro convertir esto a una clase generica en core
        """
        try:
            self.sensor_df = (
                pl.read_csv(
                    self.data_path,
                    columns=[
                        "TimeStamp",
                        "RecordDuration",
                        "FuelLevelLiters",
                        "FuelLevel",
                        "Speed",
                        "RPM",
                    ],
                )
                .with_columns(
                    pl.col("TimeStamp").str.to_datetime(),
                )
                .sort("TimeStamp")
                .with_columns(
                    pl.col("FuelLevelLiters").diff().alias("DeltaFuel"),
                    (
                        (pl.col("FuelLevelLiters") == 0)
                        & (pl.col("FuelLevelLiters").shift(-1) > 0)
                        & (pl.col("FuelLevelLiters").shift(1) > 0)
                    ).alias("SensorOff"),
                    pl.col("FuelLevel").diff().alias("DeltaFuelPercent"),
                )
            )

            self._data_loaded = True
            self._stats_cache = None
            self._stats_generated = False

        except FileNotFoundError:
            raise RuntimeError(f"Data file not found: {self.data_path}")

    def _generate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive statistics with caching"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute _load_sensor_data()")

        stats = {}
        df = self.sensor_df

        for col in df.columns:
            col_type = df.schema[col]
            col_stats = {}

            if col == "TimeStamp":  # Handle datetime separately
                col_stats = {
                    "first_record": df[col].min().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_record": df[col].max().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_duration": (df[col].max() - df[col].min()).total_seconds(),
                }
            elif col_type in [pl.Float64, pl.Int64]:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)

                col_stats = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "mode": (
                        df[col].mode().item() if not df[col].mode().is_empty() else None
                    ),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "variance": df[col].var(),
                    "std_dev": df[col].std(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": (
                        float(q3 - q1) if q1 is not None and q3 is not None else None
                    ),
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "non_null_count": df[col].len(),
                    "null_count": df[col].null_count(),
                }

            stats[col] = col_stats

        self._stats_generated = True
        self._stats_cache = stats

    def run(self):
        """Run the analysis and generate statistics"""
        self._load_sensor_data()
        self._generate_statistics()

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get the generated statistics"""
        if not self._stats_generated:
            raise RuntimeError("Primero ejecute run()")
        return self._stats_cache

    def calculate_bins(
        self, column: str, method: str = "auto", bins: int = None
    ) -> int:
        """
        Calcula el número de bins con dos modos de operación:

        1. Si se especifica `bins`: usa ese valor directamente (prioridad máxima)
        2. Si no: calcula automáticamente según el método especificado

        Args:
            column: Columna a analizar
            method: Método de cálculo (auto, fd, scott, sturges, sqrt)
            bins: Número fijo de bins deseado (opcional)
        """
        if not self._stats_generated or self._stats_cache is None:
            raise RuntimeError("Primero ejecute run()")

        if column not in self._stats_cache:
            available_cols = list(self._stats_cache.keys())
            raise ValueError(
                f"Columna '{column}' no encontrada. Disponibles: {available_cols}"
            )

        # Modo bins fijos
        if bins is not None:
            if not isinstance(bins, int) or bins <= 0:
                raise ValueError("El número de bins debe ser un entero positivo")
            return bins

        # Modo cálculo automático
        stats = self._stats_cache[column]
        n = stats.get("non_null_count", 0)

        if n == 0:
            return 0  # Caso extremo sin datos

        if method == "sqrt":
            return int(np.sqrt(n))

        if method == "sturges":
            return int(np.ceil(np.log2(n) + 1))

        if method in ["auto", "fd", "scott"]:
            iqr = stats.get("iqr", 0)
            std_dev = stats.get("std_dev", 0)
            data_range = stats["max"] - stats["min"]

            if method == "fd" or (method == "auto" and iqr > 0):
                return self._freedman_diaconis_bins(iqr, n, data_range)

            return self._scott_bins(std_dev, n, data_range)

        raise ValueError(f"Método inválido: {method}")

    def _freedman_diaconis_bins(self, iqr: float, n: int, data_range: float) -> int:
        """Cálculo de bins por Freedman-Diaconis sin restricciones"""
        if iqr == 0 or n == 0:
            return 1
        bin_width = 2 * iqr / (n ** (1 / 3))
        return int(np.ceil(data_range / bin_width)) if bin_width > 0 else 1

    def _scott_bins(self, std_dev: float, n: int, data_range: float) -> int:
        """Cálculo de bins por Scott sin restricciones"""
        if std_dev == 0 or n == 0:
            return 1
        bin_width = 3.5 * std_dev / (n ** (1 / 3))
        return int(np.ceil(data_range / bin_width)) if bin_width > 0 else 1

    def _detect_refill_events(
        self,
        min_refill_threshold=190,
        # Basado en el análisis de los documentos donde se calcula el umbral de recarga
    ) -> pl.DataFrame:

        refill_df = self.sensor_df.with_columns(
            median_next=pl.col("FuelLevelLiters").rolling_median(
                window_size=10, min_samples=3
            )
        ).with_columns(
            valid_fuel=pl.when(
                (pl.col("FuelLevelLiters") > 0)
                & (pl.col("FuelLevelLiters") < 3216)
                & (
                    (pl.col("median_next") - pl.col("FuelLevelLiters")).abs()
                    < min_refill_threshold
                )
            )
            .then(pl.col("FuelLevelLiters"))
            .otherwise(None)
            .forward_fill()
        )

        refill_df = refill_df.with_columns(
            valid_fuel=pl.when(
                (pl.col("valid_fuel") == pl.col("valid_fuel").shift(1))
                | (
                    (pl.col("valid_fuel").shift(1))
                    > (pl.col("valid_fuel") + min_refill_threshold)
                )
                | (pl.col("valid_fuel") == pl.col("valid_fuel").shift(-1))
            )
            .then(None)
            .otherwise(pl.col("valid_fuel"))
            .forward_fill(),
        )

        refill_df = (
            refill_df.with_columns(
                before_avg=pl.col("valid_fuel")
                .shift(1)  # comenzar desde el anterior registro al que se analiza
                .rolling_map(
                    window_size=10,
                    function=lambda s: (m := s.filter(s > 0).mean()) or 0.0,
                    min_samples=5,
                ),
                after_avg=pl.col("valid_fuel")
                .shift(-1)
                .reverse()
                .rolling_map(
                    window_size=10,
                    function=lambda s: (m := s.filter(s > 0).mean()) or 0.0,
                    min_samples=5,
                )
                .reverse(),
                DeltaFuel=pl.col("valid_fuel").diff().fill_null(0),
            )
            .filter(
                (pl.col("DeltaFuel") > min_refill_threshold)
                & (pl.col("after_avg") > (pl.col("before_avg") + min_refill_threshold))
            )
            .select(
                pl.col("TimeStamp"),
                pl.col("DeltaFuel"),
                pl.col("valid_fuel"),
                pl.col("after_avg"),
                pl.col("before_avg"),
            )
        )
        # Agrupar eventos cercanos
        refill_df = (
            refill_df.sort("TimeStamp")
            .with_columns(
                time_diff=pl.col("TimeStamp").diff().dt.total_seconds().fill_null(60),
            )
            .with_columns(
                group_id=pl.when(
                    pl.col("time_diff") > 10800
                )  # se agrupa datos con diferencia mayor a 6 horas
                .then(1)
                .otherwise(0)
                .cum_sum(),
            )
            .group_by("group_id")
            .agg(
                pl.when(pl.len() > 1)
                .then(pl.col("valid_fuel").last() - pl.col("valid_fuel").first())
                .otherwise(pl.col("DeltaFuel").first())
                .alias("DeltaFuel"),
                pl.col("TimeStamp").max().alias("TimeStamp"),
                pl.when(pl.len() > 1)
                .then(pl.col("valid_fuel").last())
                .otherwise(pl.col("valid_fuel").first())
                .alias("valid_fuel"),
                pl.col("before_avg").first().alias("before_avg"),
                pl.col("after_avg").last().alias("after_avg"),
            )
        )

        return refill_df.drop("group_id").sort("TimeStamp")

    def search_best_params(
        self, target: int = 341, max_combinations: int = 10000, max_threads: int = 4
    ) -> pl.DataFrame:
        """Busca los mejores parámetros para detectar al menos `target` recargas válidas"""
        print(
            f"\n🔍 Buscando combinación de parámetros para detectar al menos {target} recargas..."
        )

        # "min_refill_percentage": [round(x, 2) for x in np.arange(0.20, 0.25, 0.01)],
        # "multiplicative_factor": [round(x, 1) for x in np.arange(2.0, 2.51, 0.1)],
        # "min_fuel_percentage": list(range(20, 25)),
        # Rangos de parámetros ajustables
        param_ranges = {
            "min_refill_percentage": [round(x, 2) for x in np.arange(0.30, 0.32, 0.01)],
            "multiplicative_factor": [round(x, 1) for x in np.arange(1.0, 1.31, 0.1)],
            "min_fuel_percentage": list(range(18, 20)),
        }

        # Generar todas las combinaciones posibles
        combinations = list(
            product(
                param_ranges["min_refill_percentage"],
                param_ranges["multiplicative_factor"],
                param_ranges["min_fuel_percentage"],
            )
        )

        combinations_sorted = sorted(combinations, key=lambda x: (x[0], x[1], x[2]))[
            :max_combinations
        ]

        print(f"🔎 Probando {len(combinations_sorted)} combinaciones...")

        # Variables compartidas con seguridad para hilos
        best_lock = threading.Lock()
        best_df = None
        best_params = None

        def test_combination(combo):
            nonlocal best_df, best_params
            min_r_percent, mult_factor, min_f_percent = combo

            # Ejecutar detección con parámetros actuales
            df = self._detect_refill_events(
                min_refill_percentage=min_r_percent,
                multiplicative_factor=mult_factor,
                min_fuel_percentage=min_f_percent,
            )

            df = df.filter(
                (pl.col("TimeStamp") >= datetime(2024, 2, 1, 0, 0, 0))
                & (pl.col("TimeStamp") <= datetime(2025, 2, 28, 23, 59, 59))
            )

            # Verificar si cumple el objetivo
            if target <= df.height <= target + 3:
                with best_lock:  # Bloqueo para evitar race conditions
                    if best_df is None or df.height < best_df.height:
                        best_df = df
                        best_params = {
                            "min_refill_percentage": min_r_percent,
                            "multiplicative_factor": mult_factor,
                            "min_fuel_percentage": min_f_percent,
                        }
                return True
            return False

        with ProcessPoolExecutor(max_workers=max_threads) as executor:
            # Enviar todas las tareas al executor
            futures = [
                executor.submit(test_combination, combo)
                for combo in combinations_sorted
            ]

            # Procesar resultados conforme se completan
            for future in as_completed(futures):
                if future.result():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        if best_df is not None:
            print(f"✅ Parámetros óptimos encontrados: {best_params}")
            print(f"✅ Eventos detectados: {best_df.height}")
            best_df.write_csv("best_refill_events.csv")
            return best_df
        else:
            print("❌ No se encontraron parámetros que cumplan la condición.")
            return pl.DataFrame()


if __name__ == "__main__":

    # Inicializar analizador
    analyzer = SensorDataEDA()
    analyzer.run()

    # Configurar formato de visualización
    pl.Config.set_tbl_formatting("UTF8_FULL")
    pl.Config.set_tbl_rows(10)

    print("\n" + "=" * 50)
    print("📊 Análisis de Combustible - Terminal (Pure Polars)")
    print("=" * 50)

    # 1. Mostrar estructura del DataFrame
    print("\n🔧 Estructura del DataFrame:")
    print(f"• Registros: {analyzer.sensor_df.height}")
    print(f"• Columnas: {analyzer.sensor_df.columns}\n")

    # 2. Buscar mejores parámetros para detectar al menos 372 eventos
    # print("\n🔍 Buscando mejores parámetros para detección de recargas:")

    refill_events = analyzer._detect_refill_events()

    try:
        refill_events.write_csv("refill_events.csv")
        print("✅ Archivo CSV guardado como 'refill_events.csv'")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
        refill_events = analyzer.search_best_params(
            target=341, max_combinations=20000, max_threads=8
        )

        # 3. Mostrar resultados
        if refill_events.is_empty():
            print("❌ No se encontraron recargas válidas con los parámetros probados.")
        else:
            print(f"✅ {refill_events.height} eventos detectados")
            print(
                refill_events.select(
                    pl.col("TimeStamp").dt.strftime("%Y-%m-%d %H:%M").alias("Fecha"),
                    pl.col("DeltaFuel").round(1),
                    pl.col("valid_fuel").round(1),
                    pl.col("before_avg").round(1),
                )
            )

        # 4. Estadísticas clave
        print("\n📈 Estadísticas Clave:")
        stats = refill_events.select(
            [
                pl.col("DeltaFuel").sum().alias("Total Recargado (L)"),
                pl.col("DeltaFuel").mean().round(1).alias("Promedio/Recarga"),
                pl.col("TimeStamp").min().alias("Primera Recarga"),
                pl.col("TimeStamp").max().alias("Última Recarga"),
            ]
        )
        print(stats)
