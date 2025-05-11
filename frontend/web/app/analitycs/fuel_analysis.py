# fuel_analysis.py
import polars as pl
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from datetime import datetime, timedelta


class FuelAnalysisOptimized:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.normal_gap = 30  # Umbral fijo de 30 segundos
        self.delta_fuel_threshold = (
            0.05  # 5% del tanque como mínimo para considerar recarga
        )
        self.df_sensor = None
        self.min_date = None
        self.max_date = None

    def _load_sensor_data(self) -> pl.DataFrame:
        """Carga y prepara datos del sensor con análisis mejorado"""
        self.df_sensor = (
            pl.read_csv(
                self.base_dir / "output" / "T-210_sensor.csv",
                columns=["TimeStamp", "RecordDuration", "FuelLevelLiters"],
            )
            .sort("TimeStamp")
            .with_columns(
                pl.col("FuelLevelLiters").diff().alias("DeltaFuel"),
            )
        )

        # Calcular rango de fechas
        self.min_date = self.df_sensor["TimeStamp"].min()
        self.max_date = self.df_sensor["TimeStamp"].max()

        return self.df_sensor

    def get_temporal_gaps_stats(self) -> Dict[str, Union[float, int]]:
        """Calcula estadísticas clave de gaps y recargas"""
        df = self._load_sensor_data()

        stats = {
            "umbral_normal_gap": self.normal_gap,
            "total_registros": df.height,
            "gaps_inusuales": df["GapInusual"].sum(),
            "recargas_detectadas": df["PosibleRecargaReal"].sum(),
            "max_gap_detectado": df["RecordDuration"].max(),
            "delta_fuel_promedio": df["DeltaFuel"].mean(),
        }

        return stats

    def get_visualization_data(
        self, selected_date: datetime
    ) -> Union[pl.DataFrame, None]:
        """Devuelve datos para visualización de un día específico"""
        if self.df_with_analysis is None:
            return None

        # Calcular rango del turno (7AM a 7AM siguiente)
        start_time = selected_date.replace(hour=7, minute=0, second=0)
        end_time = start_time + timedelta(days=1)

        return self.df_with_analysis.filter(
            (pl.col("TimeStamp") >= start_time) & (pl.col("TimeStamp") < end_time)
        )

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Devuelve el rango de fechas disponible en los datos"""
        return self.min_date, self.max_date
