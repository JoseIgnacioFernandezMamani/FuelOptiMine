from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import polars as pl
from pathlib import Path
from typing import Any, Dict
from etl_core.utils.equipment_constants import TRUCK_SPECS
import numpy as np
from datetime import datetime
from typing import Optional
from etl_core.load.utils import create_client, CH_CONFIG
import logging


class SensorDataEDA:
    def __init__(self, truck_id: str) -> None:
        self.sensor_df: pl.DataFrame = pl.DataFrame()
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._data_loaded: bool = False
        self._stats_generated: bool = False
        self.truck_id = truck_id
        self.client = create_client(CH_CONFIG)

    def get_dataframe(self) -> pl.DataFrame:
        """Get the loaded DataFrame"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")
        return self.sensor_df

    def _load_sensor_data(self):
        """
        Load sensor data from clickhouse
        """
        try:

            query = f"""
            SELECT 
                TimeStamp,
                ShiftDate,
                Shift,
                TruckFleet,
                FuelLevelLiters,
                SpeedAvg,
                Acceleration,
                SlopePercent,
                ValidFuel,
                DeltaFuel,
                BeforeAvg,
                AfterAvg
            FROM xgboost_fuel
            WHERE Equipment = '{self.truck_id}'
            ORDER BY TimeStamp
            """
            # obtener los datos de clickhouse
            pandas_df = self.client.query_df(query)

            if pandas_df.empty:
                raise RuntimeError(
                    f"No se encontraron datos para el equipo {self.truck_id}"
                )

            # convertir a polars
            self.sensor_df = pl.from_pandas(pandas_df)

            # ordenar por TimeStamp
            self.sensor_df = self.sensor_df.sort("TimeStamp")

            self.sensor_df = self.sensor_df.with_columns(
                pl.col("FuelLevelLiters").diff().alias("DeltaFuel"),
                pl.col("TimeStamp").diff().dt.total_seconds().alias("TimeDiffSeconds"),
            )

            self.sensor_df = self.sensor_df.filter(pl.col("FuelLevelLiters") >= 0)

            self._data_loaded = True
            self._stats_cache = {}
            self._stats_generated = False

        except FileNotFoundError:
            raise RuntimeError(
                f"Error para cargar los datos del equipo {self.truck_id}"
            )

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
                        df[col].mode().to_list()[0]
                        if not df[col].mode().is_empty()
                        else None
                    ),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "variance": df[col].var(),
                    "std_dev": df[col].std(),
                    "p5": df[col].quantile(0.05),
                    "p10": df[col].quantile(0.10),
                    "p90": df[col].quantile(0.90),
                    "p95": df[col].quantile(0.95),
                    "p99": df[col].quantile(0.99),
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

    def close(self):
        """Close the ClickHouse client connection"""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception as e:
                logging.warning(f"Error cerrando conexión: {e}")
