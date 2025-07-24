from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, List
import polars as pl
from analytics.EDA.config.settings import DATA_DIR


class BaseAnalyzer(ABC):
    def __init__(
        self,
        data_path: str,
        datetime_col: str,
        cols: Optional[List[str]] = None,
    ):
        self.data_path = data_path
        self.datetime_col = datetime_col
        self.cols = cols or []
        self.df: Optional[pl.DataFrame] = None

    def load_data(self, columns: Optional[List[str]] = None) -> pl.DataFrame:
        """Método para cargar datos específicos"""
        try:
            self.sensor_df = pl.read_csv(
                DATA_DIR / self.data_path,
                columns=self.datetime_col + self.cols,
                try_parse_dates=True,
            ).sort(self.datetime_col)

            # Extract temporal range
            self.min_date = self.sensor_df["TimeStamp"].min()
            self.max_date = self.sensor_df["TimeStamp"].max()

            return self.sensor_df

        except FileNotFoundError:
            raise RuntimeError(f"Data file not found: {self.data_path}")

    def generate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Método común para generar estadísticas"""
        if self.df is None:
            raise RuntimeError("Datos no cargados. Llame a load_data() primero")

        stats = {}

        # Estadísticas temporales
        if self.datetime_col in self.df.columns:
            self._calculate_datetime_stats()
            stats[self.datetime_col] = self.datetime_stats

        # Estadísticas numéricas
        for col in self.numeric_cols:
            if col in self.df.columns:
                stats[col] = self._calculate_numeric_stats(col)

        return stats

    def _calculate_datetime_stats(self) -> None:
        """Calcula estadísticas para la columna temporal"""
        min_time = self.df[self.datetime_col].min()
        max_time = self.df[self.datetime_col].max()
        self.datetime_stats = {
            "first_record": min_time,
            "last_record": max_time,
            "total_duration": (max_time - min_time).total_seconds(),
        }

    def _calculate_numeric_stats(self, col: str) -> Dict[str, Any]:
        """Implementación genérica de estadísticas numéricas"""
        series = self.df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        return {
            "mean": series.mean(),
            "median": series.median(),
            "mode": series.mode().to_list() if series.mode().len() > 0 else None,
            "min": series.min(),
            "max": series.max(),
            "std_dev": series.std(),
            "variance": series.var(),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "skewness": series.skew(),
            "kurtosis": series.kurtosis(),
            "null_count": series.null_count(),
            "non_null_count": series.count(),
        }
