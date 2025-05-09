from typing import List, Type
from etl_core.transform.core.base_transformer import BaseTransformer
from .schema import SensorSchema
from typing import Optional
from pydantic import BaseModel
import polars as pl


class SensorTransformer(BaseTransformer):
    """Clase temporal para testear solo common_clean"""

    def __init__(self):
        super().__init__()

    @property
    def mandatory_columns(self) -> List[str]:
        return [
            field_name
            for field_name, field in SensorSchema.model_fields.items()
            if field.is_required()
        ]

    @property
    def schema_model(self) -> Type[BaseModel]:
        return SensorSchema

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transformaciones específicas para datos de sensores"""
        try:
            # Verificar si hay datos para procesar
            if df.is_empty():
                return df

            # Validación y preprocesamiento
            processed_df = df.sort(["Equipment", "TimeStamp"])

            # Etapa 1: Cálculos básicos
            stage1 = processed_df.with_columns(
                [
                    self._calculate_distance_traveled(),
                    self._calculate_fuel_consumption_rate("hours"),
                    self._calculate_cumulative_fuel_consumption(),
                    self._calculate_refuel_events(),
                ]
            )

            # Etapa 2: Cálculos dependientes
            stage2 = stage1.with_columns(
                [self._calculate_slope_percent(), self._calculate_efficiency_ratio()]
            )

            # Etapa 3: Clasificaciones
            return stage2.with_columns([self._categorize_slope_impact()])

        except Exception as e:
            raise ValueError(f"Unidad de tiempo no válida:")

    def _calculate_distance_traveled(self) -> pl.Expr:
        """Distancia euclidiana 3D entre puntos consecutivos en sistema local (milímetros → metros)"""
        # Convertir de milímetros a metros (dividir entre 1000)
        x = pl.col("Latitude") / 1000
        y = pl.col("Longitude") / 1000
        z = pl.col("Elevation") / 1000

        # Diferencias entre registros consecutivos por equipo
        dx = x.diff(1).over("Equipment")
        dy = y.diff(1).over("Equipment")
        dz = z.diff(1).over("Equipment")

        # Distancia 3D en metros
        distance = (dx.pow(2) + dy.pow(2) + dz.pow(2)).sqrt()

        return distance.fill_null(0).alias("distance_traveled")

    def _calculate_fuel_consumption_rate(self, time_unit: str = "hour") -> pl.Expr:
        """Consumo entre registros consecutivos (litros/hora)"""

        time_factors = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400,
            "weeks": 604800,
            "months": 2592000,
        }

        if time_unit not in time_factors:
            raise ValueError(
                f"Unidad de tiempo no válida: {time_unit}. Debe ser una de {list(time_factors.keys())}."
            )

        factor = time_factors[time_unit.lower()]

        time_diff = (
            (
                pl.col("TimeStamp").diff(1).over("Equipment").dt.total_seconds()
                / factor
            )  # calculo de diferencia de tiempo en horas
            .fill_null(0)
            .clip(lower_bound=0.1)
        )

        fuel_diff = pl.col("FuelLevelLiters").diff(1).over("Equipment")

        return (
            pl.when(fuel_diff < 0)
            .then(-fuel_diff / time_diff)
            .otherwise(0)
            .fill_null(0)
            .alias("consumption_rate")  # l/h
        )

    def _calculate_cumulative_fuel_consumption(self) -> pl.Expr:
        """Consumo historico de combustible"""
        fuel_diff = pl.col("FuelLevelLiters").diff(1).over("Equipment")

        return (
            pl.when(fuel_diff < 0)
            .then(-fuel_diff)
            .otherwise(0)
            .fill_null(0)
            .alias("fuel_consumption")
        )

    def _calculate_refuel_events(self, Threshold: int = 100) -> pl.Expr:
        """Detecta saltos significativos hacia arriba en el combustible (recargas)"""
        fuel_diff = pl.col("FuelLevelLiters").diff(1).over("Equipment")

        return (
            pl.when(fuel_diff > Threshold)
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("refuel_event")
        )

    def _calculate_slope_percent(self) -> pl.Expr:
        """Cálculo vectorizado de pendiente usando expresiones de Polars"""
        return (
            (
                pl.col("Elevation").diff(1).over("Equipment")
                / (
                    (
                        pl.col("Latitude").diff(1).over("Equipment") ** 2
                        + pl.col("Longitude").diff(1).over("Equipment") ** 2
                    ).sqrt()
                    + 1e-9
                )
                * 100
            )
            .fill_nan(0)
            .alias("slope_percent")
        )

    def _categorize_slope_impact(self) -> pl.Expr:
        """Clasificación de impacto de pendiente"""
        slope_abs = pl.col("slope_percent").abs()
        return (
            pl.when(slope_abs < 5)
            .then(pl.lit("Low"))
            .when((slope_abs >= 5) & (slope_abs < 10))
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("High"))
            .alias("slope_impact")
        )

    def _calculate_efficiency_ratio(self) -> pl.Expr:
        """Calcula la eficiencia combustible-distancia"""
        return (
            (
                pl.col("distance_traveled")
                / pl.col("fuel_consumption").clip(
                    lower_bound=0.1
                )  # Evitar división por cero
            )
            .fill_null(0)
            .alias("efficiency_ratio")
        )
