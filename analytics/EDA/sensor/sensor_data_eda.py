from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import polars as pl
from pathlib import Path
from typing import Any, Dict
from analytics.EDA.config.settings import DATA_DIR, TRUCK_SPECS
import numpy as np
from datetime import datetime
from typing import Optional


class SensorDataEDA:
    def __init__(self, truck_id: str) -> None:
        self.data_path = Path(DATA_DIR) / f"{truck_id}_sensor.csv"
        self.sensor_df: pl.DataFrame = pl.DataFrame()
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
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
            self._stats_cache = {}
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
                min_date = df.select(pl.col(col).dt.date()).min().item()
                max_date = df.select(pl.col(col).dt.date()).max().item()
                col_stats = {
                    "first_record": min_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_record": max_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_duration": (max_date - min_date).total_seconds(),
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
        return stats

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
        self, column: str, method: str = "auto", bins: Optional[int] = None
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
    ) -> pl.DataFrame:
        """Detect refill events in the sensor data for a specific truck"""
        CAPACITY = TRUCK_SPECS[truck_id]["capacity"] + 20

        # Detección y eliminación de anomalías
        refill_df = (
            self.sensor_df.with_columns(
                pl.col("FuelLevelLiters").diff(1).alias("diff_prev"),
                pl.col("FuelLevelLiters").shift(-1).diff(1).alias("diff_next"),
                pl.col("FuelLevelLiters").shift(-2).diff(1).alias("diff_next_next"),
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=100, min_samples=10)
                .alias("median_before"),
                pl.col("FuelLevelLiters")
                .shift(-100)
                .rolling_median(window_size=100, min_samples=10)
                .alias("median_after"),
            )
            .with_columns(
                (
                    (pl.col("FuelLevelLiters") >= CAPACITY)
                    | (pl.col("FuelLevelLiters") <= 0)
                    | (
                        (pl.col("diff_prev") > min_refill_threshold)
                        & (pl.col("diff_next") < -min_refill_threshold)
                    )
                    | (
                        (pl.col("diff_prev") < -min_refill_threshold)
                        & (pl.col("diff_next") > min_refill_threshold)
                    )
                    | (
                        (pl.col("diff_prev") > min_refill_threshold)
                        & (pl.col("diff_next") < -min_refill_threshold)
                        & (pl.col("diff_next_next") > min_refill_threshold)
                    )
                    | (
                        (pl.col("diff_prev") < -min_refill_threshold)
                        & (pl.col("diff_next") > min_refill_threshold)
                        & (pl.col("diff_next_next") < -min_refill_threshold)
                    )
                    | (
                        (
                            pl.col("FuelLevelLiters")
                            > pl.col("median_before") + (min_refill_threshold // 2)
                        )
                        & (
                            pl.col("FuelLevelLiters")
                            > pl.col("median_after") + min_refill_threshold
                        )
                    )
                    | (
                        (
                            pl.col("FuelLevelLiters") + min_refill_threshold
                            < pl.col("median_before")
                        )
                        & (
                            pl.col("FuelLevelLiters") + (min_refill_threshold // 2)
                            < pl.col("median_after")
                        )
                    )
                ).alias("is_anomaly")
            )
            .with_columns(
                pl.when(pl.col("is_anomaly"))
                .then(None)
                .otherwise(pl.col("FuelLevelLiters"))
                .forward_fill()
                .alias("valid_fuel")
            )
        )

        # Detección de eventos rápidos
        unfiltered_df = refill_df.with_columns(
            pl.col("valid_fuel")
            .rolling_median(window_size=15, min_samples=5)
            .alias("before_avg"),
            pl.col("valid_fuel")
            .shift(-10)
            .rolling_median(window_size=15, min_samples=5)
            .alias("after_avg"),
            pl.col("valid_fuel").diff().fill_null(0).alias("delta_fuel"),
        )

        refill_df = unfiltered_df.filter(
            (pl.col("delta_fuel") > min_refill_threshold - 25)
            & (pl.col("after_avg") > (pl.col("before_avg") + min_refill_threshold))
        ).sort("TimeStamp")

        # Procesamiento de anomalías
        anomalies_df = unfiltered_df.with_columns(
            pl.col("is_anomaly")
            .cast(pl.Int8)
            .diff()
            .fill_null(1)
            .ne(0)
            .cum_sum()
            .alias("anomaly_group"),
            pl.col("FuelLevelLiters").diff().fill_null(0).alias("delta_fuel_anomaly"),
            pl.col("valid_fuel")
            .shift(-100)
            .rolling_median(window_size=50, min_samples=20)
            .alias("improved_after_avg_100"),
            pl.col("valid_fuel")
            .shift(-50)
            .rolling_median(window_size=30, min_samples=15)
            .alias("improved_after_avg_50"),
            pl.col("valid_fuel")
            .shift(-25)
            .rolling_median(window_size=20, min_samples=10)
            .alias("improved_after_avg_25"),
        ).with_columns(
            pl.when(
                (pl.col("improved_after_avg_100").is_not_null())
                & (pl.col("improved_after_avg_100") <= CAPACITY)
            )
            .then(pl.col("improved_after_avg_100"))
            .when(
                (pl.col("improved_after_avg_50").is_not_null())
                & (pl.col("improved_after_avg_50") <= CAPACITY)
            )
            .then(pl.col("improved_after_avg_50"))
            .when(
                (pl.col("improved_after_avg_25").is_not_null())
                & (pl.col("improved_after_avg_25") <= CAPACITY)
            )
            .then(pl.col("improved_after_avg_25"))
            .otherwise(None)
            .alias("best_improved_after_avg")
        )

        # Candidatos para mejora
        improved_after_candidates = (
            anomalies_df.filter(
                (pl.col("best_improved_after_avg").is_not_null())
                & (pl.col("best_improved_after_avg") > 2000)
                & (pl.col("best_improved_after_avg") <= CAPACITY + 100)
            )
            .group_by("anomaly_group")
            .agg(
                pl.col("TimeStamp").first().alias("candidate_timestamp"),
                pl.col("best_improved_after_avg")
                .max()
                .alias("best_improved_after_avg"),
            )
            .sort("candidate_timestamp")
        )

        # Agrupación de eventos continuos
        refill_df = (
            refill_df.with_columns(
                pl.col("TimeStamp")
                .diff()
                .dt.total_seconds()
                .fill_null(60)
                .alias("time_diff")
            )
            .with_columns(
                pl.when(pl.col("time_diff") > 10800)
                .then(1)
                .otherwise(0)
                .cum_sum()
                .alias("group_id")
            )
            .group_by("group_id")
            .agg(
                (pl.col("after_avg").last() - pl.col("before_avg").first()).alias(
                    "delta_fuel"
                ),
                pl.col("TimeStamp").max().alias("TimeStamp"),
                pl.when(pl.len() > 1)
                .then(pl.col("valid_fuel").last())
                .otherwise(pl.col("valid_fuel").first())
                .alias("valid_fuel"),
                pl.col("before_avg").first().alias("before_avg"),
                pl.col("after_avg").last().alias("after_avg"),
            )
        )

        # Corrección de delta_fuel con aux_df
        aux_df = (
            unfiltered_df.with_columns(
                pl.col("valid_fuel")
                .shift(-25)
                .rolling_median(window_size=15, min_samples=5)
                .alias("after_aux"),
                pl.col("valid_fuel")
                .shift(1)
                .rolling_median(window_size=15, min_samples=5)
                .alias("before_aux"),
            )
            .with_columns(
                (pl.col("after_aux") - pl.col("before_aux")).alias("calculated_delta")
            )
            .select(["TimeStamp", "after_aux", "before_aux", "calculated_delta"])
        )

        refill_df = refill_df.join(aux_df, on="TimeStamp", how="left").with_columns(
            pl.when(
                (pl.col("delta_fuel") < 500)
                & (pl.col("after_aux") > pl.col("before_aux") + 500)
            )
            .then(pl.col("calculated_delta"))
            .otherwise(pl.col("delta_fuel"))
            .alias("delta_fuel")
        )

        # Encontrar candidatos cercanos
        joined_df_candidates = (
            refill_df.join(improved_after_candidates, how="cross")
            .with_columns(
                (pl.col("TimeStamp") - pl.col("candidate_timestamp"))
                .dt.total_seconds()
                .alias("time_diff_candidate")
            )
            .filter(
                (pl.col("time_diff_candidate") >= 0)
                & (pl.col("time_diff_candidate") <= 14400)
            )
        )

        nearest_candidates = (
            joined_df_candidates.sort("time_diff_candidate")
            .group_by("TimeStamp")
            .agg(
                pl.col("candidate_timestamp")
                .first()
                .alias("nearest_candidate_timestamp"),
                pl.col("best_improved_after_avg")
                .first()
                .alias("candidate_improved_after_avg"),
            )
        )

        # Unión y mejora de valores
        refill_df = (
            refill_df.join(nearest_candidates, on="TimeStamp", how="left")
            .with_columns(
                pl.when(
                    (pl.col("candidate_improved_after_avg").is_not_null())
                    & (pl.col("candidate_improved_after_avg") > pl.col("after_avg"))
                    & (
                        pl.col("candidate_improved_after_avg") - pl.col("after_avg")
                        >= 200
                    )
                )
                .then(pl.col("candidate_improved_after_avg"))
                .otherwise(pl.col("after_avg"))
                .alias("final_after_avg")
            )
            .with_columns(
                (pl.col("final_after_avg") - pl.col("before_avg")).alias(
                    "final_delta_fuel"
                ),
                pl.col("TimeStamp").alias("final_TimeStamp"),
            )
        )

        # Resultado final
        return (
            refill_df.filter(pl.col("final_delta_fuel") > 500)
            .select(
                [
                    pl.col("final_TimeStamp").alias("TimeStamp"),
                    "valid_fuel",
                    pl.col("final_delta_fuel").alias("delta_fuel"),
                    "before_avg",
                    pl.col("final_after_avg").alias("after_avg"),
                ]
            )
            .sort("TimeStamp")
        )


import os

if __name__ == "__main__":

    os.makedirs("refill_test", exist_ok=True)

    print("\n" + "=" * 50)
    print("📊 Análisis de Combustible - Por Camión (Pure Polars)")
    print("=" * 50)

    # 1. Filtrar y procesar cada camión individualmente
    TRUCKS: set[str] = {"T-220"}
    # agregacion manual
    manual_refill: pl.DataFrame = pl.DataFrame(
        {
            "truck_id": [
                "T-215",
                "T-216",
                "T-214",
                "T-218",
                "T-221",
                "T-224",
                "T-236",
                "T-242",
                "T-211",
                "T-218",
                "T-218",
                "T-218",
                "T-218",
                "T-218",
                "T-219",
            ],
            "TimeStamp": [
                datetime(2024, 6, 13, 1, 19, 0),  # recarga constante
                datetime(2024, 6, 11, 17, 9, 0),  # recarga constante
                datetime(2024, 2, 17, 11, 58, 30),  # recarga pequenia
                datetime(2024, 6, 10, 21, 50, 30),  # recarga constante
                datetime(2024, 5, 23, 13, 54, 30),  # recarga constante
                datetime(2024, 6, 4, 16, 2, 30),  # recarga constante
                datetime(2024, 2, 17, 9, 55, 30),  # recarga constante
                datetime(2024, 6, 4, 0, 29, 0),  # recarga constante
                datetime(2024, 4, 8, 16, 59, 0),  # recarga pequenia
                datetime(2025, 1, 26, 6, 5, 1),  # recarga pequenia
                datetime(2025, 2, 2, 21, 9, 30),  # recarga como ruido
                datetime(2025, 2, 7, 21, 12, 0),  # recarga como ruido
                datetime(2025, 2, 8, 17, 26, 2),  # recarga como ruido
                datetime(2025, 2, 26, 20, 12, 0),  # recarga como ruido
                datetime(2024, 3, 1, 9, 13, 30),  # recarga pequenia
            ],
            "valid_fuel": [
                2841.0,
                3124.48,
                3216.0,
                3061.12,
                2896.96,
                2882.24,
                4375.77,
                4375.77,
                3180.8,
                2253.76,
                2887.36,
                3000.96,
                2416.96,
                2872.32,
                3180.8,
            ],
            "delta_fuel": [
                2009.8,
                2195.08,
                144.32,
                1725.12,
                1970.24,
                2281.92,
                2480.474,
                3284.22,
                535.04,
                701.12,
                1823.68,
                2267.2,
                1509.28,
                2067.84,
                504.96,
            ],
            "before_avg": [
                830.2,
                953.92,
                3071.68,
                1336.0,
                926.72,
                600.32,
                1895.296,
                1091.548,
                2645.76,
                1552.64,
                1063.68,
                733.76,
                907.84,
                804.48,
                2675.84,
            ],
            "after_avg": [
                2840.0,
                3149.00,
                3216.0,
                3061.12,
                2896.96,
                2882.24,
                4375.77,
                4375.77,
                3180.8,
                2253.76,
                2887.36,
                3000.96,
                2416.96,
                2872.32,
                3180.8,
            ],
        }
    )

    for truck_id in TRUCKS:

        truck_analyzer = SensorDataEDA(truck_id=truck_id)
        truck_analyzer.run()

        # 2. Detectar eventos de recarga
        try:
            refill_events: pl.DataFrame = truck_analyzer._detect_refill_events()
            # Filtrar manuales por camión actual
            refill = manual_refill.filter(pl.col("truck_id") == truck_id)

            # Si hay recargas manuales para este camión, las concatenamos
            if refill.height > 0:
                refill_events = pl.concat(
                    [refill_events, refill.drop("truck_id")]
                ).sort("TimeStamp")
            # 3. Guardar en archivo individual
            filename = f"frontend/web/app/refill/{truck_id}_refill_events.csv"
            refill_events.write_csv(filename)
            print(f"✅ {refill_events.height} eventos guardados en {filename}")

        except Exception as e:
            print(f"error {e}")

    print("\n✅ Proceso completado para todos los camiones!")
