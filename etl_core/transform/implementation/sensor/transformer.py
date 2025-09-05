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


class SensorTransformer(BaseTransformer):
    """Optimized transformer for sensor data using Polars expressions"""

    def __init__(self, truck_id) -> None:
        super().__init__()
        self.metrics.update(
            {
                "outliers_removed": 0,
                "invalid_geo_records": 0,
                "categorical_null_empty_replaced": 0,
                "refill_events_detected": 0,
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
        """Optimized transformation pipeline using minimal with_columns calls"""

        # 1. Basic transformations in one operation
        df = self._apply_all_basic_transformations(df)

        # 2. Count categorical null/empty values replaced
        self.metrics["categorical_null_empty_replaced"] = (
            count_null_empty_categorical_values(df, self.categorical_columns)
        )

        # 3. Apply filters and update metrics
        df = self._apply_filters_with_metrics(df)

        # 4. Sort once for all subsequent operations
        df = df.sort("ShiftDate", "TimeStamp")

        # 5. Apply enrichment calculations (distance, slope, etc.)
        df = self._apply_complete_enrichment(df)

        # 6. Apply refill detection and fuel consumption
        df = self._apply_refill_and_consumption_detection(df)

        # 7. Update final metrics
        self._update_final_metrics(df)

        # 8. Return only final schema columns that exist
        final_columns = list(SensorSchema.model_fields.keys()) + [
            "SpeedAvg",
            "Acceleration",
            "DistanceTraveled",
            "SlopePercent",
            "BeforeMedian",
            "AfterMedian",
            "FuelConsumption",
            "ValidFuel",
            "DeltaFuel",
            "BeforeAvg",
            "AfterAvg",
        ]
        available_columns = [col for col in final_columns if col in df.columns]
        return df.select(available_columns)

    def _apply_all_basic_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all basic transformations in a single with_columns operation"""
        conversion_exprs = get_coordinate_conversion_exprs()
        categorical_exprs = get_categorical_normalization_exprs(
            self.categorical_columns
        )

        # Combine all basic transformations
        all_exprs = (
            conversion_exprs
            + categorical_exprs
            + [
                # Outlier handling
                pl.when(pl.col("FuelLevelLiters") > 4500)
                .then(None)
                .otherwise(pl.col("FuelLevelLiters"))
                .alias("FuelLevelLiters"),
                pl.when(pl.col("Speed") > 60)
                .then(None)
                .otherwise(pl.col("Speed"))
                .alias("Speed"),
            ]
        )

        return df.with_columns(all_exprs)

    def _apply_filters_with_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filters and calculate metrics in one operation"""
        before_records = df.height

        geo_validation_expr = get_geo_validation_expr()
        df_filtered = df.filter(
            pl.col("FuelLevelLiters").is_not_null()
            & pl.col("Speed").is_not_null()
            & geo_validation_expr
        )

        # Calculate metrics
        after_outliers = df.filter(
            pl.col("FuelLevelLiters").is_not_null() & pl.col("Speed").is_not_null()
        ).height

        self.metrics["outliers_removed"] = before_records - after_outliers
        self.metrics["invalid_geo_records"] = after_outliers - df_filtered.height

        return df_filtered

    def _apply_complete_enrichment(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all enrichment calculations with minimal with_columns calls"""

        df = df.with_columns(
            [
                # Speed smoothing
                pl.col("Speed")
                .rolling_mean(window_size=5, center=True, min_samples=3)
                .alias("SpeedAvg"),
            ]
        )

        # Step 2: Delta calculations and haversine distance
        df = df.with_columns(
            [
                # Acceleration
                (
                    (pl.col("SpeedAvg") - pl.col("SpeedAvg").shift(1))
                    * 1000
                    / 3600
                    / pl.col("RecordDuration")
                )
                .fill_null(0)
                .alias("Acceleration"),
            ]
        )

        # Step 3: Distance calculations and route elements
        df = df.with_columns(
            [
                # MRUV distance
                (
                    (pl.col("SpeedAvg") * 1000 / 3600 * pl.col("RecordDuration"))
                    + (0.5 * pl.col("Acceleration") * pl.col("RecordDuration").pow(2))
                )
                .fill_null(0)
                .alias("DistanceTraveled"),
                (pl.col("Elevation").diff().fill_null(0)).alias("ElevationDelta"),
            ]
        )

        df = df.with_columns(
            [
                # Slope calculation
                (
                    (
                        pl.col("ElevationDelta")
                        / pl.col("DistanceTraveled").clip(lower_bound=0.1)
                    )
                    * 100
                )
                .fill_nan(0)
                .fill_null(0)
                .alias("SlopePercent"),
            ]
        )

        # Step 7: Final validation and cleanup
        df = df.with_columns(
            [
                # Validated slope
                pl.when(
                    (pl.col("RecordDuration").cast(pl.Float64) <= 86400)
                    & (pl.col("SlopePercent") >= -20)
                    & (pl.col("SlopePercent") <= 20)
                )
                .then(pl.col("SlopePercent"))
                .otherwise(None)
                .alias("SlopePercent"),
                # Validated distance
                pl.when(
                    (pl.col("DistanceTraveled") > 0)
                    & (pl.col("DistanceTraveled") < 1500)
                    & (pl.col("RecordDuration") <= 86400)
                )
                .then(pl.col("DistanceTraveled"))
                .otherwise(None)
                .alias("DistanceTraveled"),
            ]
        )

        return df

    def _apply_refill_and_consumption_detection(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply refill detection and fuel consumption calculation with minimal with_columns"""

        # Detect refill events
        refill_events = self._detect_refill_events(df)
        self.metrics["refill_events_detected"] = refill_events.height

        # Join with main dataframe and add fuel consumption logic
        df = df.join(refill_events, on="TimeStamp", how="left")

        # Apply all fuel-related calculations in minimal steps
        df = df.with_columns(
            [
                # Rolling median for fuel smoothing
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=50, min_samples=5)
                .alias("MedianBefore"),
            ]
        )

        df = df.with_columns(
            [
                # Auxiliary column for fuel consumption detection
                pl.when(
                    (
                        (pl.col("FuelLevelLiters").shift(1) < pl.col("FuelLevelLiters"))
                        & (
                            pl.col("FuelLevelLiters").shift(-1)
                            < pl.col("FuelLevelLiters")
                        )
                    )
                    | (
                        (pl.col("FuelLevelLiters").shift(1) > pl.col("FuelLevelLiters"))
                        & (
                            pl.col("FuelLevelLiters").shift(-1)
                            > pl.col("FuelLevelLiters")
                        )
                    )
                )
                .then(pl.col("FuelLevelLiters"))
                .otherwise(None)
                .forward_fill()
                .alias("AuxFuel")
            ]
        )

        # Final fuel consumption calculation
        df = df.with_columns(
            [
                pl.when(
                    (pl.col("AuxFuel").shift(1) > pl.col("AuxFuel").shift(-1))
                    & (pl.col("AuxFuel").shift(1) > pl.col("AuxFuel"))
                    & (pl.col("AuxFuel").shift(-1) > pl.col("AuxFuel"))
                    & (pl.col("MedianBefore") > pl.col("AuxFuel").shift(-1))
                    & (pl.col("MedianBefore") > pl.col("AuxFuel").shift(1))
                )
                .then(pl.col("AuxFuel").shift(1) - pl.col("AuxFuel").shift(-1))
                .otherwise(None)
                .alias("FuelConsumption")
            ]
        )

        # Drop auxiliary column
        if "AuxFuel" in df.columns:
            df = df.drop("AuxFuel")

        return df

    def _update_final_metrics(self, df: pl.DataFrame) -> None:
        """Update final metrics"""
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )

    def _detect_refill_events(
        self, sensor_df: pl.DataFrame, min_refill_threshold=190
    ) -> pl.DataFrame:
        """Detect fuel refill events with optimized column operations"""

        CAPACITY = TRUCK_SPECS[self.truck_id]["capacity"] + 100

        # Step 1: All anomaly detection calculations in one operation
        refill_df = sensor_df.with_columns(
            [
                # Difference calculations
                pl.col("FuelLevelLiters").diff(1).alias("DiffPrev"),
                pl.col("FuelLevelLiters").shift(-1).diff(1).alias("DiffNext"),
                pl.col("FuelLevelLiters").shift(-2).diff(1).alias("DiffNextNext"),
                # Rolling medians
                pl.col("FuelLevelLiters")
                .rolling_median(window_size=100, min_samples=10)
                .alias("MedianBefore"),
                pl.col("FuelLevelLiters")
                .shift(-100)
                .rolling_median(window_size=100, min_samples=10)
                .alias("MedianAfter"),
            ]
        )

        # Step 2: Anomaly detection and valid fuel calculation
        refill_df = refill_df.with_columns(
            [
                # Complex anomaly detection in one expression
                (
                    (pl.col("FuelLevelLiters") >= CAPACITY)
                    | (pl.col("FuelLevelLiters") <= 0)
                    | (
                        (pl.col("DiffPrev") > min_refill_threshold)
                        & (pl.col("DiffNext") < -min_refill_threshold)
                    )
                    | (
                        (pl.col("DiffPrev") < -min_refill_threshold)
                        & (pl.col("DiffNext") > min_refill_threshold)
                    )
                    | (
                        (pl.col("DiffPrev") > min_refill_threshold)
                        & (pl.col("DiffNext") < -min_refill_threshold)
                        & (pl.col("DiffNextNext") > min_refill_threshold)
                    )
                    | (
                        (pl.col("DiffPrev") < -min_refill_threshold)
                        & (pl.col("DiffNext") > min_refill_threshold)
                        & (pl.col("DiffNextNext") < -min_refill_threshold)
                    )
                    | (
                        (
                            pl.col("FuelLevelLiters")
                            > pl.col("MedianBefore") + (min_refill_threshold // 2)
                        )
                        & (
                            pl.col("FuelLevelLiters")
                            > pl.col("MedianAfter") + min_refill_threshold
                        )
                    )
                    | (
                        (
                            pl.col("FuelLevelLiters") + min_refill_threshold
                            < pl.col("MedianBefore")
                        )
                        & (
                            pl.col("FuelLevelLiters") + (min_refill_threshold // 2)
                            < pl.col("MedianAfter")
                        )
                    )
                ).alias("IsAnomaly"),
            ]
        )

        # Step 3: Valid fuel and refill detection calculations
        refill_df = refill_df.with_columns(
            [
                # Valid fuel (replace anomalies with forward fill)
                pl.when(pl.col("IsAnomaly"))
                .then(None)
                .otherwise(pl.col("FuelLevelLiters"))
                .forward_fill()
                .alias("ValidFuel"),
            ]
        )

        # Step 4: All refill detection calculations
        unfiltered_df = refill_df.with_columns(
            [
                # Before and after averages
                pl.col("ValidFuel")
                .rolling_median(window_size=15, min_samples=5)
                .alias("BeforeAvg"),
                pl.col("ValidFuel")
                .shift(-10)
                .rolling_median(window_size=15, min_samples=5)
                .alias("AfterAvg"),
                # Delta fuel
                pl.col("ValidFuel").diff().fill_null(0).alias("DeltaFuel"),
                # Improved after averages
                pl.col("ValidFuel")
                .shift(-100)
                .rolling_median(window_size=50, min_samples=20)
                .alias("ImprovedAfterAvg100"),
                pl.col("ValidFuel")
                .shift(-50)
                .rolling_median(window_size=30, min_samples=15)
                .alias("ImprovedAfterAvg50"),
            ]
        )

        # Filter for refill candidates
        refill_df = unfiltered_df.filter(
            (pl.col("DeltaFuel") > min_refill_threshold - 25)
            & (pl.col("AfterAvg") > (pl.col("BeforeAvg") + min_refill_threshold))
        ).sort("TimeStamp")

        # Final calculations and grouping
        refill_df = refill_df.with_columns(
            [
                # Improved after average selection
                pl.when(
                    (pl.col("ImprovedAfterAvg50").is_not_null())
                    & (pl.col("ImprovedAfterAvg50") <= CAPACITY)
                    & (pl.col("ValidFuel") < pl.col("ImprovedAfterAvg50"))
                )
                .then(pl.col("ImprovedAfterAvg50"))
                .when(
                    (pl.col("ImprovedAfterAvg100").is_not_null())
                    & (pl.col("ImprovedAfterAvg100") <= CAPACITY)
                    & (pl.col("ValidFuel") < pl.col("ImprovedAfterAvg100"))
                )
                .then(pl.col("ImprovedAfterAvg100"))
                .otherwise(pl.col("AfterAvg"))
                .alias("AfterAvg"),
            ]
        )

        # Recalculate delta and group events
        refill_df = refill_df.with_columns(
            [
                (pl.col("AfterAvg") - pl.col("BeforeAvg")).abs().alias("DeltaFuel"),
                pl.col("TimeStamp")
                .diff()
                .dt.total_seconds()
                .fill_null(60)
                .alias("TimeDiff"),
            ]
        )

        refill_df = refill_df.with_columns(
            [
                pl.when(pl.col("TimeDiff") > 10800)
                .then(1)
                .otherwise(0)
                .cum_sum()
                .alias("GroupId")
            ]
        )

        # Group and aggregate
        result_df = (
            refill_df.group_by("GroupId")
            .agg(
                [
                    (pl.col("AfterAvg").last() - pl.col("BeforeAvg").first()).alias(
                        "DeltaFuel"
                    ),
                    pl.col("TimeStamp").max().alias("TimeStamp"),
                    pl.when(pl.len() > 1)
                    .then(pl.col("ValidFuel").last())
                    .otherwise(pl.col("ValidFuel").first())
                    .alias("ValidFuel"),
                    pl.col("BeforeAvg").first().alias("BeforeAvg"),
                    pl.col("AfterAvg").last().alias("AfterAvg"),
                ]
            )
            .filter(pl.col("DeltaFuel") > 500)
            .select(["TimeStamp", "ValidFuel", "DeltaFuel", "BeforeAvg", "AfterAvg"])
            .sort("TimeStamp")
        )

        return result_df
