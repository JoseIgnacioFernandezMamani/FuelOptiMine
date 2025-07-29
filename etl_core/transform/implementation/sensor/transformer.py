from typing import List, Type, Optional
from etl_core.transform.core import BaseTransformer
from etl_core.utils.sensor_schemas import SensorSchema
from etl_core.transform.utils import (
    get_coordinate_conversion_exprs,
    get_geo_validation_expr,
    get_categorical_normalization_exprs,
    count_null_empty_categorical_values,
)
from pydantic import BaseModel
import polars as pl
from polars import Expr
import math


class SensorTransformer(BaseTransformer):
    """Optimized transformer for sensor data using Polars expressions"""

    def __init__(self) -> None:
        super().__init__()
        self.metrics.update(
            {
                "outliers_removed": 0,
                "invalid_geo_records": 0,
                "categorical_null_empty_replaced": 0,
            }
        )

        self.categorical_columns = [
            "Shift",
            "FuelGauge",
            "Ralenti",
            "TruckFleet",
        ]

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

    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Optimized transformation pipeline using Polars expressions"""

        # 1. Pipeline de transformaciones básicas en una sola operación
        df = self._apply_basic_transformations(df)

        # 2. Métrica: contar categorías vacías reemplazadas
        self.metrics["categorical_null_empty_replaced"] = (
            count_null_empty_categorical_values(df, self.categorical_columns)
        )

        # 3. Filtrado y métricas
        df = self._apply_filters(df)

        # 4. Enriquecimiento: cálculos de distancia y pendiente
        df = self._apply_enrichment_calculations(df)

        # 5. Métricas finales
        self._update_final_metrics(df)

        # 6. Retornar solo las columnas del schema final que existan
        final_columns: list[str] = list(SensorSchema.model_fields.keys()) + [
            "DistanceTraveled",
            "SlopePercent",
        ]
        available_columns: list[str] = [
            col for col in final_columns if col in df.columns
        ]
        return df.select(available_columns)

    def _apply_basic_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplicar todas las transformaciones básicas en una sola operación"""
        conversion_exprs: list[Expr] = get_coordinate_conversion_exprs()
        categorical_exprs: list[Expr] = get_categorical_normalization_exprs(
            self.categorical_columns
        )
        outlier_exprs: list[Expr] = self._get_outlier_handling_exprs()

        return df.with_columns(conversion_exprs + categorical_exprs + outlier_exprs)

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplicar filtros de outliers y coordenadas inválidas"""
        before_outliers: int = df.height

        # Filtrar outliers y registros geográficos inválidos en una sola operación
        geo_validation_expr: Expr = get_geo_validation_expr()
        df_filtered: pl.DataFrame = df.filter(
            pl.col("FuelLevelLiters").is_not_null()
            & pl.col("Speed").is_not_null()
            & geo_validation_expr
        )

        # Calcular métricas de filtrado
        after_outliers: int = df_filtered.filter(
            pl.col("FuelLevelLiters").is_not_null() & pl.col("Speed").is_not_null()
        ).height

        self.metrics["outliers_removed"] = before_outliers - after_outliers
        self.metrics["invalid_geo_records"] = after_outliers - df_filtered.height

        return df_filtered

    def _apply_enrichment_calculations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplicar cálculos de enriquecimiento de manera optimizada"""
        # Ordenar datos por fecha y timestamp
        df = df.sort("ShiftDate", "TimeStamp")

        # Aplicar transformaciones paso a paso para evitar dependencias circulares
        df = df.with_columns(self._get_haversine_intermediate_exprs())
        df = df.with_columns(self._get_delta_exprs())
        df = df.with_columns(self._calculate_distance_traveled_expr())
        df = df.with_columns(self._calculate_slope_percent_expr())
        df = df.with_columns(self._get_final_validation_exprs())

        # Eliminar columnas intermedias (solo si existen)
        columns_to_drop: list[str] = [
            col
            for col in [
                "LatitudeRad",
                "LongitudeRad",
                "LatitudeRadPrev",
                "LongitudeRadPrev",
                "delta_lat",
                "delta_lon",
            ]
            if col in df.columns
        ]
        if columns_to_drop:
            df = df.drop(columns_to_drop)

        return df

    def _get_outlier_handling_exprs(self) -> List[pl.Expr]:
        """Expresiones para manejar outliers de manera optimizada"""
        return [
            pl.when(pl.col("FuelLevelLiters") > 4500)
            .then(None)
            .otherwise(pl.col("FuelLevelLiters"))
            .alias("FuelLevelLiters"),
            pl.when(pl.col("Speed") > 60)
            .then(None)
            .otherwise(pl.col("Speed"))
            .alias("Speed"),
        ]

    def _get_haversine_intermediate_exprs(self) -> List[pl.Expr]:
        """Columnas necesarias en radianes para Haversine - optimizado"""
        deg_to_rad: Expr = pl.lit(math.pi / 180)
        return [
            (pl.col("Latitude") * deg_to_rad).alias("LatitudeRad"),
            (pl.col("Longitude") * deg_to_rad).alias("LongitudeRad"),
            (pl.col("Latitude").shift(1).over("Equipment") * deg_to_rad).alias(
                "LatitudeRadPrev"
            ),
            (pl.col("Longitude").shift(1).over("Equipment") * deg_to_rad).alias(
                "LongitudeRadPrev"
            ),
        ]

    def _get_delta_exprs(self) -> List[pl.Expr]:
        """Columnas delta para Haversine"""
        return [
            (pl.col("LatitudeRad") - pl.col("LatitudeRadPrev")).alias("delta_lat"),
            (pl.col("LongitudeRad") - pl.col("LongitudeRadPrev")).alias("delta_lon"),
        ]

    def _calculate_distance_traveled_expr(self) -> pl.Expr:
        """Cálculo optimizado de distancia entre puntos con fórmula Haversine"""
        R: Expr = pl.lit(6371000.0)  # Radio terrestre en metros como literal

        # Fórmula Haversine optimizada
        a: Expr = (
            (pl.col("delta_lat") / 2).sin().pow(2)
            + pl.col("LatitudeRadPrev").cos()
            * pl.col("LatitudeRad").cos()
            * (pl.col("delta_lon") / 2).sin().pow(2)
        ).clip(lower_bound=0.0, upper_bound=1.0)

        c: Expr = 2 * pl.arctan2(a.sqrt(), (1 - a).sqrt())

        return (R * c).fill_null(0).alias("DistanceTraveled")

    def _calculate_slope_percent_expr(self) -> pl.Expr:
        """Cálculo optimizado de pendiente en %"""
        elevation_diff: Expr = pl.col("Elevation").diff(1).over("Equipment")
        slope: Expr = (elevation_diff / pl.col("DistanceTraveled")) * 100

        return slope.fill_nan(0).fill_null(0).alias("SlopePercent")

    def _get_final_validation_exprs(self) -> List[pl.Expr]:
        """Expresiones de validación final para distancia y pendiente"""
        return [
            pl.when(
                (pl.col("RecordDuration").cast(pl.Float64) <= 60)
                & (pl.col("SlopePercent") >= -20)
                & (pl.col("SlopePercent") <= 20)
            )
            .then(pl.col("SlopePercent"))
            .otherwise(None)
            .alias("SlopePercent"),
            pl.when(
                (pl.col("DistanceTraveled") > 0)
                & (pl.col("RecordDuration") <= 60)
                & (pl.col("SlopePercent").is_not_null())
            )
            .then(pl.col("DistanceTraveled"))
            .otherwise(None)
            .alias("DistanceTraveled"),
        ]

    def _update_final_metrics(self, df: pl.DataFrame) -> None:
        """Actualizar métricas finales"""
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )
