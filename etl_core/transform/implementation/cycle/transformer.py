from typing import List, Type, Optional
from etl_core.transform.core import BaseTransformer
from etl_core.utils.cycle_schemas import CycleSchema
from etl_core.transform.utils import (
    get_coordinate_conversion_exprs,
)
from pydantic import BaseModel
import polars as pl
from polars import Expr


class CycleTransformer(BaseTransformer):
    """Optimized transformer for mining cycle data using Polars expressions"""

    def __init__(self) -> None:
        super().__init__()
        self.metrics.update(
            {
                "outliers_removed": 0,
                "invalid_geo_records": 0,
                "categorical_empty_fixed": 0,
                "negative_times_fixed": 0,
                "invalid_tonnage_fixed": 0,
                "datetime_parsing_errors": 0,
                "expanded_records": 0,
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
        conversion_exprsG: list[Expr] = get_coordinate_conversion_exprs(
            lat_col="G_Latitude", lon_col="G_Longitude", elev_col="G_Elevation"
        )
        conversion_exprsD: list[Expr] = get_coordinate_conversion_exprs(
            lat_col="D_Latitude", lon_col="D_Longitude", elev_col="D_Elevation"
        )
        categorical_exprs: list[Expr] = self._get_categorical_normalization_exprs(df)
        outlier_exprs: list[Expr] = self._get_outlier_handling_exprs()
        time_validation_exprs: list[Expr] = self._get_time_validation_exprs()
        tonnage_validation_exprs: list[Expr] = self._get_tonnage_validation_exprs()

        # 2. Apply all transformations in a single step
        df = df.with_columns(
            conversion_exprsG
            + conversion_exprsD
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
        df = self._apply_filters(df)

        # 6. Sort by shiftdate and Equipment before expansion
        df = df.sort("E_TravelingStart")

        # 7. Expand cycles into stages
        df = self._expand_cycles_to_stages(df)

        # 8. Update final metrics
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )

        return df

    def _expand_cycles_to_stages(self, df: pl.DataFrame) -> pl.DataFrame:
        """Expand each cycle record into 8 stage records using vectorized operations"""

        # Add a unique cycle identifier to maintain order
        df = df.with_row_count("cycle_id")

        # Create stage mappings as lists for vectorized processing
        stage_types = [
            "empty_traveling",
            "waiting_empty",
            "spotting_empty",
            "material_loading",
            "load_travel",
            "load_waiting_queue",
            "load_positioning",
            "material_unloading",
        ]
        stage_sequences = [1, 2, 3, 4, 5, 6, 7, 8]
        duration_cols = [
            "TravelingEmpty",
            "WaitingEmpty",
            "SpottingEmpty",
            "LoadingMaterial",
            "Hauling",
            "WaitingLoad",
            "SpottingLoad",
            "UnloadingMaterial",
        ]
        end_cols = [
            "E_TravelingEnd",
            "E_WaitingEnd",
            "E_SpottingEnd",
            "E_LoadingEnd",
            "L_HaulingEnd",
            "L_WaitingEnd",
            "L_SpottingEnd",
            "L_UnloadingEnd",
        ]
        categories = [
            "empty",
            "empty",
            "empty",
            "loading",
            "loaded",
            "loaded",
            "loaded",
            "unloading",
        ]

        # Create expanded DataFrame using explode
        df_expanded = df.with_columns(
            [
                pl.lit(stage_types).alias("StageType"),
                pl.lit(stage_sequences).alias("StageSequence"),
                pl.lit(duration_cols).alias("duration_col_name"),
                pl.lit(end_cols).alias("end_col_name"),
                pl.lit(categories).alias("category"),
            ]
        ).explode(
            [
                "StageType",
                "StageSequence",
                "duration_col_name",
                "end_col_name",
                "category",
            ]
        )

        # Create RecordDuration and TimeStamp using dynamic column selection
        df_expanded = df_expanded.with_columns(
            [
                # Dynamic duration selection
                pl.when(pl.col("duration_col_name") == "TravelingEmpty")
                .then(pl.col("TravelingEmpty"))
                .when(pl.col("duration_col_name") == "WaitingEmpty")
                .then(pl.col("WaitingEmpty"))
                .when(pl.col("duration_col_name") == "SpottingEmpty")
                .then(pl.col("SpottingEmpty"))
                .when(pl.col("duration_col_name") == "LoadingMaterial")
                .then(pl.col("LoadingMaterial"))
                .when(pl.col("duration_col_name") == "Hauling")
                .then(pl.col("Hauling"))
                .when(pl.col("duration_col_name") == "WaitingLoad")
                .then(pl.col("WaitingLoad"))
                .when(pl.col("duration_col_name") == "SpottingLoad")
                .then(pl.col("SpottingLoad"))
                .when(pl.col("duration_col_name") == "UnloadingMaterial")
                .then(pl.col("UnloadingMaterial"))
                .alias("RecordDuration"),
                # Dynamic timestamp selection
                pl.when(pl.col("end_col_name") == "E_TravelingEnd")
                .then(pl.col("E_TravelingEnd"))
                .when(pl.col("end_col_name") == "E_WaitingEnd")
                .then(pl.col("E_WaitingEnd"))
                .when(pl.col("end_col_name") == "E_SpottingEnd")
                .then(pl.col("E_SpottingEnd"))
                .when(pl.col("end_col_name") == "E_LoadingEnd")
                .then(pl.col("E_LoadingEnd"))
                .when(pl.col("end_col_name") == "L_HaulingEnd")
                .then(pl.col("L_HaulingEnd"))
                .when(pl.col("end_col_name") == "L_WaitingEnd")
                .then(pl.col("L_WaitingEnd"))
                .when(pl.col("end_col_name") == "L_SpottingEnd")
                .then(pl.col("L_SpottingEnd"))
                .when(pl.col("end_col_name") == "L_UnloadingEnd")
                .then(pl.col("L_UnloadingEnd"))
                .alias("TimeStamp"),
            ]
        )

        # Apply conditional logic for each category using vectorized operations
        df_result = df_expanded.with_columns(
            [
                # LoadingZone logic
                pl.when(pl.col("category").is_in(["empty", "loading"]))
                .then(pl.col("LoadingZone"))
                .otherwise(pl.lit(None))
                .alias("LoadingZone_final"),
                # Material logic
                pl.when(pl.col("category").is_in(["loading", "loaded", "unloading"]))
                .then(pl.col("Material"))
                .otherwise(pl.lit(None))
                .alias("Material_final"),
                # Tonnage logic
                pl.when(pl.col("category").is_in(["loading", "loaded", "unloading"]))
                .then(pl.col("MeasuredTonnage"))
                .otherwise(pl.lit(None))
                .alias("MeasuredTonnage_final"),
                pl.when(pl.col("category").is_in(["loading", "loaded", "unloading"]))
                .then(pl.col("ReportedTonnage"))
                .otherwise(pl.lit(None))
                .alias("ReportedTonnage_final"),
                # Destination logic
                pl.when(pl.col("category").is_in(["loading", "loaded", "unloading"]))
                .then(pl.col("DestinationType"))
                .otherwise(pl.lit(None))
                .alias("DestinationType_final"),
                pl.when(pl.col("category").is_in(["loading", "loaded", "unloading"]))
                .then(pl.col("Destination"))
                .otherwise(pl.lit(None))
                .alias("Destination_final"),
                # Distance logic
                pl.when(pl.col("category") == "loading")
                .then(pl.col("DistanceEmpty"))
                .when(pl.col("category") == "unloading")
                .then(pl.col("DistanceLoaded"))
                .otherwise(pl.lit(None))
                .alias("Distance"),
                # Coordinates logic
                pl.when(pl.col("category").is_in(["loading"]))
                .then(pl.col("G_Latitude"))
                .when(pl.col("category").is_in(["unloading"]))
                .then(pl.col("D_Latitude"))
                .otherwise(pl.lit(None))
                .alias("Latitude"),
                pl.when(pl.col("category").is_in(["loading"]))
                .then(pl.col("G_Longitude"))
                .when(pl.col("category").is_in(["unloading"]))
                .then(pl.col("D_Longitude"))
                .otherwise(pl.lit(None))
                .alias("Longitude"),
                pl.when(pl.col("category").is_in(["loading"]))
                .then(pl.col("G_Elevation"))
                .when(pl.col("category").is_in(["unloading"]))
                .then(pl.col("D_Elevation"))
                .otherwise(pl.lit(None))
                .alias("Elevation"),
                # Derived fields
                pl.when(pl.col("category") == "unloading")
                .then(pl.col("MeasuredTonnage") / pl.col("RecordDuration"))
                .otherwise(pl.lit(None))
                .alias("TonnageEfficiency"),
                pl.when(pl.col("category") == "loading")
                .then(pl.col("DistanceEmpty") / pl.col("RecordDuration"))
                .when(pl.col("category") == "unloading")
                .then(pl.col("DistanceLoaded") / pl.col("RecordDuration"))
                .otherwise(pl.lit(None))
                .alias("AverageSpeed"),
                ((pl.col("RecordDuration") / pl.col("TotalCycleTime")) * 100).alias(
                    "TimeEfficiencyPercentage"
                ),
            ]
        )

        final_df = df_result.sort("cycle_id", "StageSequence")
        # Select final columns and filter valid records
        final_df = final_df.select(
            [
                "ShiftDate",
                "Shift",
                "Shovel",
                "ShovelModel",
                "Equipment",
                "TruckFleet",
                "StageType",
                "StageSequence",
                "RecordDuration",
                "TimeStamp",
                pl.col("LoadingZone_final").alias("LoadingZone"),
                pl.col("Material_final").alias("Material"),
                pl.col("MeasuredTonnage_final").alias("MeasuredTonnage"),
                pl.col("ReportedTonnage_final").alias("ReportedTonnage"),
                pl.col("DestinationType_final").alias("DestinationType"),
                pl.col("Destination_final").alias("Destination"),
                "Distance",
                "Latitude",
                "Longitude",
                "Elevation",
                "TonnageEfficiency",
                "AverageSpeed",
                "TimeEfficiencyPercentage",
                "cycle_id",
            ]
        )

        self.metrics["expanded_records"] = final_df.height
        return final_df

    ## antiguo
    def _get_categorical_normalization_exprs(self, df: pl.DataFrame) -> list[pl.Expr]:
        """Expressions for normalizing categorical fields"""
        categorical_columns: list[str] = [
            "Shift",
            "Shovel",
            "ShovelModel",
            "TruckFleet",
            "LoadingZone",
            "Material",
            "DestinationType",
            "Destination",
        ]
        # Filtrar solo columnas que existen en el DataFrame
        valid_columns: list[str] = [
            col for col in categorical_columns if col in df.columns
        ]

        return [
            pl.when(
                pl.col(col).is_null() | (pl.col(col) == "") | (pl.col(col) == "NaN")
            )
            .then(pl.lit("NaN"))
            .otherwise(pl.col(col))
            .alias(col)
            for col in valid_columns
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
                (pl.col("TotalCycleTime") < 0) | (pl.col("TotalCycleTime") > 43200)
            )  # 1440 minutes = 24 hours
            .then(None)
            .otherwise(pl.col("TotalCycleTime"))
            .alias("TotalCycleTime"),
        ]

    def _get_time_validation_exprs(self) -> list[pl.Expr]:
        """Expressions for validating and fixing time-based fields"""
        time_columns: list[str] = [
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
            # Fix outliers and negative values in tonnage fields
            pl.when((pl.col("MeasuredTonnage") < 0) | (pl.col("MeasuredTonnage") > 500))
            .then(0.0)
            .otherwise(pl.col("MeasuredTonnage"))
            .alias("MeasuredTonnage"),
            pl.when((pl.col("ReportedTonnage") < 0) | (pl.col("ReportedTonnage") > 500))
            .then(0.0)
            .otherwise(pl.col("ReportedTonnage"))
            .alias("ReportedTonnage"),
        ]

    def _count_categorical_fixes(self, df: pl.DataFrame) -> None:
        """Count fixed empty values in categorical fields"""
        categorical_columns: list[str] = [
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
                null_count: int = df.filter(
                    pl.col(col).is_in(["", "NaN"]) | pl.col(col).is_null()
                ).height
                self.metrics["categorical_empty_fixed"] += null_count

    def _count_time_fixes(self, df: pl.DataFrame) -> None:
        """Count negative time values that were fixed"""
        time_columns: list[str] = [
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
                negative_count: int = df.filter(pl.col(col) < 0).height
                self.metrics["negative_times_fixed"] += negative_count

    def _count_tonnage_fixes(self, df: pl.DataFrame) -> None:
        """Count invalid tonnage values that were fixed"""
        tonnage_columns: list[str] = ["MeasuredTonnage", "ReportedTonnage"]
        for col in tonnage_columns:
            if col in df.columns:
                # Only count negative values (values >500 are converted to null)
                negative_count: int = df.filter(pl.col(col) < 0).height
                self.metrics["invalid_tonnage_fixed"] += negative_count

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all filters and update metrics"""
        # Improved coordinate validation
        geo_validation: Expr = self._get_geo_validation_expr(
            "G_"
        ) & self._get_geo_validation_expr("D_")

        # 1. Filter records with invalid coordinates
        before_geo: int = df.height
        df = df.filter(geo_validation)
        self.metrics["invalid_geo_records"] = before_geo - df.height

        # 2. Filter outliers (values converted to null)
        before_outliers: int = df.height
        outlier_conditions: Expr = (
            ((pl.col("DistanceEmpty") >= 0) & (pl.col("DistanceEmpty") < 15000))
            & ((pl.col("DistanceLoaded") >= 0) & (pl.col("DistanceLoaded") < 15000))
            & (
                (pl.col("EquivalentDistance") >= 0)
                & (pl.col("EquivalentDistance") < 15000)
            )
            & ((pl.col("MeasuredTonnage") >= 0) & (pl.col("MeasuredTonnage") < 230))
            & ((pl.col("ReportedTonnage") >= 0) & (pl.col("ReportedTonnage") < 230))
        )

        df = df.filter(outlier_conditions)
        self.metrics["outliers_removed"] = before_outliers - df.height

        return df

    def _get_geo_validation_expr(self, prefix: str) -> pl.Expr:
        """Expression to validate coordinates with prefix"""
        return (
            pl.col(f"{prefix}Latitude").is_between(-90, 90)
            & pl.col(f"{prefix}Longitude").is_between(-180, 180)
            & pl.col(f"{prefix}Elevation").is_between(-1000, 8000)
        )

    # delete after columns
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
                # Calculate loading time percentage
                pl.when(pl.col("TotalCycleTime") > 0)
                .then((pl.col("LoadingMaterial") / pl.col("TotalCycleTime")) * 100)
                .otherwise(0.0)
                .alias("LoadingTimePercentage"),
                # Calculate percentage of hauling time
                pl.when(pl.col("TotalCycleTime") > 0)
                .then((pl.col("Hauling") / pl.col("TotalCycleTime")) * 100)
                .otherwise(0.0)
                .alias("HaulingTimePercentage"),
            ]
        )
