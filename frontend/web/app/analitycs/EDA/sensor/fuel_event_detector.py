import polars as pl
from pathlib import Path
from typing import Any, Dict
from analitycs.EDA.config.settings import DATA_DIR
import numpy as np


class FuelEventDetector:
    def __init__(self) -> None:
        self.data_path = Path(DATA_DIR) / "T-210_sensor.csv"
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
                        "Speed",
                        "RPM",
                    ],
                    try_parse_dates=True,
                )
                .sort("TimeStamp")
                .with_columns(
                    pl.col("FuelLevelLiters").diff().alias("DeltaFuel"),
                )
            )

            self._data_loaded = True
            self._stats_cache = None
            self._stats_generated = False

        except FileNotFoundError:
            raise RuntimeError(f"Data file not found: {self.data_path}")
