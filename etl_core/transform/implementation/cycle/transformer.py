from typing import List, Type, Optional
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.cycle_schemas import CycleSchema
from etl_core.transform.utils.unit_converter import (
    get_coordinate_conversion_exprs,
    get_geo_validation_expr,
)
from pydantic import BaseModel
import polars as pl


class CycleTransformer(BaseTransformer):
    """Optimized transformer for mining cycle data using Polars expressions"""

    def __init__(self):
        super().__init__()
        self.metrics.update(
            {
                "outliers_removed": 0,
                "invalid_geo_records": 0,
                "categorical_empty_fixed": 0,
                "negative_times_fixed": 0,
                "invalid_tonnage_fixed": 0,
                "datetime_parsing_errors": 0,
            }
        )

    @property
    def mandatory_columns(self) -> List[str]:
        """Dynamically get mandatory columns from Pydantic schema"""
        return [
            field_name
            for field_name, field in CycleSchema.model_fields.items()
            if field.is_required()
        ]

    @property
    def schema_model(self) -> Type[BaseModel]:
        """Pydantic schema for data validation"""
        return CycleSchema

    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Optimized transformation pipeline using expressions"""

        # 1. Collect all transformation expressions
        conversion_exprs = get_coordinate_conversion_exprs()
        categorical_exprs = self._get_categorical_normalization_exprs()
        outlier_exprs = self._get_outlier_handling_exprs()
        time_validation_exprs = self._get_time_validation_exprs()
        tonnage_validation_exprs = self._get_tonnage_validation_exprs()
        geo_validation_expr = get_geo_validation_expr()

        # 2. Apply all transformations in a single step
        df = df.with_columns(
            conversion_exprs
            + categorical_exprs
            + outlier_exprs
            + time_validation_exprs
            + tonnage_validation_exprs
        )

        # 3. Count fixes for categorical values
        self._count_categorical_fixes(df)

        # 4. Count time and tonnage fixes
        self._count_time_fixes(df)
        self._count_tonnage_fixes(df)

        # 5. Apply filters and update metrics
        df = self._apply_filters(df, geo_validation_expr)

        # 6. Calculate derived fields
        df = self._calculate_derived_fields(df)

        # 7. Update final metrics
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )

        # 8. Sort by ShiftDate and Equipment
        df = df.sort("ShiftDate", "Equipment")

        return df

    def _get_categorical_normalization_exprs(self) -> list[pl.Expr]:
        """Expressions for normalizing categorical fields"""
        categorical_columns = [
            "Shift",
            "Shovel",
            "ShovelModel",
            "TruckFleet",
            "LoadingZone",
            "Material",
            "DestinationType",
            "Destination",
        ]
        return [
            pl.when(
                pl.col(col).is_null() | (pl.col(col) == "") | (pl.col(col) == "NaN")
            )
            .then(pl.lit("NaN"))
            .otherwise(pl.col(col))
            .alias(col)
            for col in categorical_columns
            if col in df.columns
        ]

    def _get_outlier_handling_exprs(self) -> list[pl.Expr]:
        """Expressions for handling outlier values in distances and times"""
        return [
            # Distance outliers (convert unrealistic distances to null)
            pl.when((pl.col("DistanceEmpty") < 0) | (pl.col("DistanceEmpty") > 50000))
            .then(None)
            .otherwise(pl.col("DistanceEmpty"))
            .alias("DistanceEmpty"),
            pl.when((pl.col("DistanceLoaded") < 0) | (pl.col("DistanceLoaded") > 50000))
            .then(None)
            .otherwise(pl.col("DistanceLoaded"))
            .alias("DistanceLoaded"),
            pl.when(
                (pl.col("EquivalentDistance") < 0)
                | (pl.col("EquivalentDistance") > 100000)
            )
            .then(None)
            .otherwise(pl.col("EquivalentDistance"))
            .alias("EquivalentDistance"),
            # Total cycle time outliers (more than 24 hours seems unrealistic)
            pl.when(
                (pl.col("TotalCycleTime") < 0) | (pl.col("TotalCycleTime") > 1440)
            )  # 1440 minutes = 24 hours
            .then(None)
            .otherwise(pl.col("TotalCycleTime"))
            .alias("TotalCycleTime"),
        ]

    def _get_time_validation_exprs(self) -> list[pl.Expr]:
        """Expressions for validating and fixing time-based fields"""
        time_columns = [
            "TravelingEmpty",
            "WaitingEmpty",
            "SpottingEmpty",
            "LoadingMaterial",
            "Hauling",
            "WaitingLoad",
            "SpottingLoad",
            "UnloadingMaterial",
        ]
        return [
            pl.when(pl.col(col) < 0).then(pl.lit(0.0)).otherwise(pl.col(col)).alias(col)
            for col in time_columns
        ]

    def _get_tonnage_validation_exprs(self) -> list[pl.Expr]:
        """Expressions for validating tonnage fields"""
        return [
            # Fix negative tonnage values
            pl.when(pl.col("MeasuredTonnage") < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("MeasuredTonnage"))
            .alias("MeasuredTonnage"),
            pl.when(pl.col("ReportedTonnage") < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("ReportedTonnage"))
            .alias("ReportedTonnage"),
            # Cap extremely high tonnage values (assuming max truck capacity ~400 tons)
            pl.when(pl.col("MeasuredTonnage") > 500)
            .then(None)
            .otherwise(pl.col("MeasuredTonnage"))
            .alias("MeasuredTonnage"),
            pl.when(pl.col("ReportedTonnage") > 500)
            .then(None)
            .otherwise(pl.col("ReportedTonnage"))
            .alias("ReportedTonnage"),
        ]

    def _count_categorical_fixes(self, df: pl.DataFrame) -> None:
        """Count fixed empty values in categorical fields"""
        categorical_columns = [
            "Shift",
            "Shovel",
            "ShovelModel",
            "TruckFleet",
            "LoadingZone",
            "Material",
            "DestinationType",
            "Destination",
        ]
        for col in categorical_columns:
            if col in df.columns:
                null_count = df.filter(
                    pl.col(col).is_null() | (pl.col(col) == "") | (pl.col(col) == "NaN")
                ).height
                self.metrics["categorical_empty_fixed"] += null_count

    def _count_time_fixes(self, df: pl.DataFrame) -> None:
        """Count negative time values that were fixed"""
        time_columns = [
            "TravelingEmpty",
            "WaitingEmpty",
            "SpottingEmpty",
            "LoadingMaterial",
            "Hauling",
            "WaitingLoad",
            "SpottingLoad",
            "UnloadingMaterial",
        ]
        for col in time_columns:
            if col in df.columns:
                negative_count = df.filter(pl.col(col) < 0).height
                self.metrics["negative_times_fixed"] += negative_count

    def _count_tonnage_fixes(self, df: pl.DataFrame) -> None:
        """Count invalid tonnage values that were fixed"""
        tonnage_columns = ["MeasuredTonnage", "ReportedTonnage"]
        for col in tonnage_columns:
            if col in df.columns:
                invalid_count = df.filter(
                    (pl.col(col) < 0) | (pl.col(col) > 500)
                ).height
                self.metrics["invalid_tonnage_fixed"] += invalid_count

    def _apply_filters(
        self, df: pl.DataFrame, geo_validation_expr: pl.Expr
    ) -> pl.DataFrame:
        """Apply all filters and update metrics"""

        # Filter outliers (values converted to null)
        before_outliers = df.height
        outlier_conditions = (
            pl.col("DistanceEmpty").is_not_null()
            & pl.col("DistanceLoaded").is_not_null()
            & pl.col("EquivalentDistance").is_not_null()
            & pl.col("TotalCycleTime").is_not_null()
            & pl.col("MeasuredTonnage").is_not_null()
            & pl.col("ReportedTonnage").is_not_null()
        )
        df = df.filter(outlier_conditions)
        self.metrics["outliers_removed"] = before_outliers - df.height

        # Filter invalid geo records (both origin and destination coordinates)
        before_geo = df.height
        # Create validation expressions for both G_ (origin) and D_ (destination) coordinates
        geo_validation_g = pl.struct(
            [pl.col("G_Latitude"), pl.col("G_Longitude"), pl.col("G_Elevation")]
        ).map_elements(lambda x: self._validate_coordinates(x), return_dtype=pl.Boolean)

        geo_validation_d = pl.struct(
            [pl.col("D_Latitude"), pl.col("D_Longitude"), pl.col("D_Elevation")]
        ).map_elements(lambda x: self._validate_coordinates(x), return_dtype=pl.Boolean)

        df = df.filter(geo_validation_g & geo_validation_d)
        self.metrics["invalid_geo_records"] = before_geo - df.height

        return df

    def _validate_coordinates(self, coord_struct) -> bool:
        """Validate coordinate values are within reasonable ranges"""
        lat, lon, elev = coord_struct
        if lat is None or lon is None or elev is None:
            return False
        # Basic coordinate validation (adjust ranges based on your mining location)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180 and -1000 <= elev <= 8000):
            return False
        return True

    def _calculate_derived_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate derived fields like efficiency metrics"""
        return df.with_columns(
            [
                # Calculate total distance
                (pl.col("DistanceEmpty") + pl.col("DistanceLoaded")).alias(
                    "TotalDistance"
                ),
                # Calculate tonnage efficiency (tons per minute)
                pl.when(pl.col("TotalCycleTime") > 0)
                .then(pl.col("MeasuredTonnage") / pl.col("TotalCycleTime"))
                .otherwise(0.0)
                .alias("TonnageEfficiency"),
                # Calculate speed (distance per minute)
                pl.when(pl.col("TotalCycleTime") > 0)
                .then(
                    (pl.col("DistanceEmpty") + pl.col("DistanceLoaded"))
                    / pl.col("TotalCycleTime")
                )
                .otherwise(0.0)
                .alias("AverageSpeed"),
                # Calculate loading efficiency percentage
                pl.when(pl.col("TotalCycleTime") > 0)
                .then((pl.col("LoadingMaterial") / pl.col("TotalCycleTime")) * 100)
                .otherwise(0.0)
                .alias("LoadingTimePercentage"),
                # Calculate hauling efficiency percentage
                pl.when(pl.col("TotalCycleTime") > 0)
                .then((pl.col("Hauling") / pl.col("TotalCycleTime")) * 100)
                .otherwise(0.0)
                .alias("HaulingTimePercentage"),
            ]
        )
