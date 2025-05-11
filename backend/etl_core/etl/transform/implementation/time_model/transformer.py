from typing import List, Type
from etl_core.etl.transform.core.base_transformer import BaseTransformer
from .schema import TimeModelSchema
from pydantic import BaseModel
import polars as pl


class TimeModelTransformer(BaseTransformer):
    """Transformador para modelado temporal de operaciones mineras"""

    def __init__(self):
        super().__init__()
        self._window_sizes = [60, 300, 900]  # Ventanas en segundos: 1, 5, 15 min

    @property
    def mandatory_columns(self) -> List[str]:
        return ["Equipment", "TimeStamp"]

    @property
    def schema_model(self) -> Type[BaseModel]:
        return TimeModelSchema

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ejecuta el pipeline de transformación temporal"""
        try:
            if df.is_empty():
                return df

            # Preparación de datos temporal
            processed_df = self._preprocess_data(df)

            # Etapa 1: Características básicas de tiempo
            stage1 = processed_df.with_columns(
                [
                    self._extract_time_features(),
                    self._calculate_time_since_last_event(),
                    self._calculate_operating_session(),
                ]
            )

            # Etapa 2: Agregaciones temporales
            stage2 = stage1.with_columns(
                [
                    *self._create_rolling_features("FuelLevelLiters"),
                    *self._create_rolling_features("RPM"),
                    self._calculate_time_based_efficiency(),
                ]
            )

            # Etapa 3: Características para modelado
            return stage2.with_columns(
                [
                    self._calculate_equipment_utilization(),
                    self._predict_remaining_operating_time(),
                    self._detect_temporal_anomalies(),
                ]
            )

        except Exception as e:
            raise RuntimeError(f"Error en modelado temporal: {str(e)}")

    def _preprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Prepara los datos para análisis temporal"""
        return df.sort(["Equipment", "TimeStamp"]).with_columns(
            [
                pl.col("TimeStamp").dt.cast_time_unit("ms"),
                pl.col("FuelLevelLiters").interpolate().over("Equipment"),
            ]
        )

    def _extract_time_features(self) -> pl.Expr:
        """Extrae características temporales básicas"""
        return pl.struct(
            [
                pl.col("TimeStamp").dt.hour().alias("hour"),
                pl.col("TimeStamp").dt.minute().alias("minute"),
                (pl.col("TimeStamp").dt.minute() // 15).alias("quarter_hour"),
                pl.col("TimeStamp").dt.weekday().alias("weekday"),
                pl.col("TimeStamp").dt.day().alias("day_of_month"),
            ]
        ).alias("time_features")

    def _calculate_time_since_last_event(self) -> pl.Expr:
        """Calcula tiempo desde último evento por tipo"""
        return (
            pl.col("TimeStamp")
            .diff()
            .over(["Equipment", "EventType"])
            .dt.total_seconds()
            .alias("time_since_last_event")
        )

    def _calculate_operating_session(self) -> pl.Expr:
        """Identifica sesiones operativas continuas"""
        return (
            pl.when(pl.col("Status").is_in(["Operativo", "Producción"]))
            .then(pl.col("time_since_last_event") < 300)
            .otherwise(False)
            .cumsum()
            .over("Equipment")
            .alias("operating_session")
        )

    def _create_rolling_features(self, metric: str) -> List[pl.Expr]:
        """Crea características móviles para métricas clave"""
        return [
            pl.col(metric)
            .rolling_mean(window_size, min_periods=1)
            .over("Equipment")
            .alias(f"{metric}_rolling_{window_size}s")
            for window_size in self._window_sizes
        ]

    def _calculate_time_based_efficiency(self) -> pl.Expr:
        """Eficiencia operativa con ventana temporal"""
        return (
            pl.col("distance_traveled").rolling_sum(900)
            / pl.col("FuelLevelLiters").rolling_sum(900).clip(lower_bound=0.1)
        ).alias("efficiency_15min")

    def _calculate_equipment_utilization(self) -> pl.Expr:
        """Calcula utilización real del equipo"""
        operating_time = (
            pl.col("Status").is_in(["Producción", "Transporte"]).cast(pl.Int64)
        )
        return (operating_time.rolling_mean(3600).over("Equipment") * 100).alias(
            "utilization_pct"
        )

    def _predict_remaining_operating_time(self) -> pl.Expr:
        """Predice tiempo restante de operación basado en consumo"""
        return (
            (
                pl.col("FuelLevelLiters")
                / pl.col("FuelLevelLiters")
                .diff()
                .abs()
                .rolling_mean(300)
                .over("Equipment")
            )
            .clip(lower_bound=0)
            .alias("predicted_remaining_time")
        )

    def _detect_temporal_anomalies(self) -> pl.Expr:
        """Detección de anomalías temporales"""
        return pl.when(
            (pl.col("RPM_rolling_60s") < 500)
            & (pl.col("utilization_pct") > 80)
            .then("Low RPM High Utilization")
            .when(
                (pl.col("FuelLevelLiters_rolling_300s") < 0.1)
                & (pl.col("distance_traveled") > 100)
                .then("Fuel Anomaly")
                .otherwise("Normal")
                .alias("temporal_anomaly")
            )
        )
