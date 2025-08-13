from typing import List, Type, Optional
from etl_core.transform.core import BaseTransformer
from etl_core.utils.sensor_schemas import SensorSchema
from etl_core.transform.utils import (
    get_coordinate_conversion_exprs,
    get_geo_validation_expr,
    get_categorical_normalization_exprs,
    count_null_empty_categorical_values,
)
from etl_core.utils.equipment_constants import TRUCK_SPECS

from pydantic import BaseModel
import polars as pl
from polars import Expr
import math


## aplicarle la deteccion de eventos de recarga y finalmente el consumo estimado
class SensorTransformer(BaseTransformer):
    """Optimized transformer for sensor data using Polars expressions"""

    def __init__(self, truck_id) -> None:
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
        self.truck_id = truck_id

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

        # 5. Detección de eventos de recarga (último paso)
        df = self._apply_refill_detection(df)

        # 6. Métricas finales
        self._update_final_metrics(df)

        # 7. Retornar solo las columnas del schema final que existan
        final_columns: list[str] = list(SensorSchema.model_fields.keys()) + [
            "DistanceTraveled",
            "SlopePercent",
            "before_median",
            "after_median",
            "fuel_consumption",
            "valid_fuel",
            "delta_fuel",
            "before_avg",
            "after_avg",
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

    def _apply_refill_detection(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplicar detección de eventos de recarga y agregar columnas al DataFrame principal"""

        df = df.with_columns(
            [
                # Mediana móvil antes (ventana de 15 registros hacia atrás)
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=15, min_samples=5)
                .over("Equipment")
                .alias("before_median"),
                # Mediana móvil después (ventana de 15 registros hacia adelante)
                pl.col("FuelLevelLiters")
                .shift(-10)
                .rolling_median(window_size=15, min_samples=5)
                .over("Equipment")
                .alias("after_median"),
            ]
        )

        # stage 2
        # 2. Calcular consumo de combustible con las condiciones especificadas
        df = df.with_columns(
            [
                pl.when(
                    # Condiciones: before > after AND diferencia < 190
                    (pl.col("before_median") > pl.col("after_median"))
                    & ((pl.col("before_median") - pl.col("after_median")) < 190)
                )
                .then(pl.col("before_median") - pl.col("after_median"))
                .otherwise(None)
                .alias("fuel_consumption")
            ]
        )

        # 1. Detectar eventos de recarga usando la función existente
        refill_events = self._detect_refill_events(df)

        # 2. Actualizar métrica de eventos detectados
        self.metrics["refill_events_detected"] = refill_events.height

        # 3. Hacer join con el DataFrame principal para agregar las columnas de recarga
        # Left join por TimeStamp para que todos los registros se mantengan
        df = df.join(refill_events, on="TimeStamp", how="left")

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

    # This space is reserved for adding additional methods in the future.

    def _detect_refill_events(
        self, sensor_df: pl.DataFrame, min_refill_threshold=190
    ) -> pl.DataFrame:
        """
        Detects fuel refill events from raw fuel level sensor data for a specific truck.

        The function:
        - removes anomalies (spikes, valleys, plateaus, out-of-range values),
        - computes before/after rolling medians,
        - detects refill candidates,
        - groups nearby records into single refill events,
        - and returns significant refill events with the original/group TimeStamp kept.
        """

        CAPACITY: float | int = (
            TRUCK_SPECS[self.truck_id]["capacity"] + 100
        )  # keep your original +100 tolerance

        # --- 1) Detect and mask anomalies (keep data sorted by TimeStamp) ---
        refill_df: pl.DataFrame = (
            sensor_df.with_columns(
                # columns used only for removal of spikes and valleys
                pl.col("FuelLevelLiters").diff(1).alias("diff_prev"),
                pl.col("FuelLevelLiters").shift(-1).diff(1).alias("diff_next"),
                pl.col("FuelLevelLiters").shift(-2).diff(1).alias("diff_next_next"),
                # columns used only for removal of valleys and plateaus
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=100, min_samples=10)
                .alias("median_before"),
                pl.col("FuelLevelLiters")
                .shift(-100)
                .rolling_median(window_size=100, min_samples=10)
                .alias("median_after"),
            ).with_columns(
                (
                    # Out of range values
                    (pl.col("FuelLevelLiters") >= CAPACITY)
                    | (pl.col("FuelLevelLiters") <= 0)
                    # Sudden spike up then down
                    | (
                        (pl.col("diff_prev") > min_refill_threshold)
                        & (pl.col("diff_next") < -min_refill_threshold)
                    )
                    # Sudden spike down then up
                    | (
                        (pl.col("diff_prev") < -min_refill_threshold)
                        & (pl.col("diff_next") > min_refill_threshold)
                    )
                    # Complex spike pattern (up-down-up)
                    | (
                        (pl.col("diff_prev") > min_refill_threshold)
                        & (pl.col("diff_next") < -min_refill_threshold)
                        & (pl.col("diff_next_next") > min_refill_threshold)
                    )
                    # Complex valley pattern (down-up-down)
                    | (
                        (pl.col("diff_prev") < -min_refill_threshold)
                        & (pl.col("diff_next") > min_refill_threshold)
                        & (pl.col("diff_next_next") < -min_refill_threshold)
                    )
                    # Plateau anomaly: small rise at start, larger drop at end
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
                    # Valley anomaly: sharp drop at start, smaller rise at end
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
            # Replace anomalies with last valid reading
            .with_columns(
                pl.when(pl.col("is_anomaly"))
                .then(None)
                .otherwise(pl.col("FuelLevelLiters"))
                .forward_fill()
                .alias("valid_fuel")
            )
        )

        # --- 2) Rolling medians / deltas to detect refill candidates ---
        unfiltered_df: pl.DataFrame = refill_df.with_columns(
            pl.col("valid_fuel")
            .rolling_median(window_size=15, min_samples=5)
            .alias("before_avg"),
            pl.col("valid_fuel")
            .shift(-10)
            .rolling_median(window_size=15, min_samples=5)
            .alias("after_avg"),
            pl.col("valid_fuel").diff().fill_null(0).alias("delta_fuel"),
            pl.col("valid_fuel")
            .shift(-100)
            .rolling_median(window_size=50, min_samples=20)
            .alias("improved_after_avg_100"),
            pl.col("valid_fuel")
            .shift(-50)
            .rolling_median(window_size=30, min_samples=15)
            .alias("improved_after_avg_50"),
        )

        # --- 3) First-pass: fast refill detection ---
        refill_df = unfiltered_df.filter(
            (pl.col("delta_fuel") > min_refill_threshold - 25)
            & (pl.col("after_avg") > (pl.col("before_avg") + min_refill_threshold))
        ).sort("TimeStamp")

        # get truth after average
        refill_df = refill_df.with_columns(
            pl.when(
                (pl.col("improved_after_avg_50").is_not_null())
                & (pl.col("improved_after_avg_50") <= CAPACITY)
                & (pl.col("valid_fuel") < pl.col("improved_after_avg_50"))
            )
            .then(pl.col("improved_after_avg_50"))
            .when(
                (pl.col("improved_after_avg_100").is_not_null())
                & (pl.col("improved_after_avg_100") <= CAPACITY)
                & (pl.col("valid_fuel") < pl.col("improved_after_avg_100"))
            )
            .then(pl.col("improved_after_avg_100"))
            .otherwise(pl.col("after_avg"))
            .alias("after_avg")
        ).with_columns(
            (pl.col("after_avg") - pl.col("before_avg")).abs().alias("delta_fuel")
        )

        # Grouping of continuous events
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

        # result final
        return (
            refill_df.filter(pl.col("delta_fuel") > 500)
            .select(
                [
                    "TimeStamp",
                    "valid_fuel",
                    "delta_fuel",
                    "before_avg",
                    "after_avg",
                ]
            )
            .sort("TimeStamp")
        )
