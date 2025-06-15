from typing import List, Type
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.time_model_schemas import TimeModelSchema
from pydantic import BaseModel
import polars as pl


from typing import List, Type, Optional
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.time_model_schemas import TimeModelSchema
from pydantic import BaseModel
import polars as pl


class TimeModelTransformer(BaseTransformer):
    """Optimized transformer for mining time model data using Polars expressions"""

    def __init__(self):
        super().__init__()
        self.metrics.update(
            {
                "outliers_removed": 0,
                "categorical_empty_fixed": 0,
                "negative_durations_fixed": 0,
                "duplicate_events_removed": 0,
                "timestamp_sequence_errors": 0,
                "invalid_status_combinations": 0,
            }
        )

    @property
    def mandatory_columns(self) -> List[str]:
        """Dynamically get mandatory columns from Pydantic schema"""
        return [
            field_name
            for field_name, field in TimeModelSchema.model_fields.items()
            if field.is_required()
        ]

    @property
    def schema_model(self) -> Type[BaseModel]:
        """Pydantic schema for data validation"""
        return TimeModelSchema

    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Optimized transformation pipeline using expressions"""

        # 1. Collect all transformation expressions
        categorical_exprs = self._get_categorical_normalization_exprs()
        duration_validation_exprs = self._get_duration_validation_exprs()
        status_normalization_exprs = self._get_status_normalization_exprs()

        # 2. Apply all transformations in a single step
        df = df.with_columns(
            categorical_exprs + duration_validation_exprs + status_normalization_exprs
        )

        # 3. Count fixes for categorical values
        self._count_categorical_fixes(df)

        # 4. Count duration fixes
        self._count_duration_fixes(df)

        # 5. Remove duplicates and validate sequences
        df = self._remove_duplicate_events(df)
        df = self._validate_timestamp_sequences(df)

        # 6. Apply filters and update metrics
        df = self._apply_filters(df)

        # 7. Calculate derived fields for time modeling
        df = self._calculate_time_model_fields(df)

        # 8. Update final metrics
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )

        # 9. Sort by Equipment, ShiftDate, and TimeStamp for time model analysis
        df = df.sort("Equipment", "ShiftDate", "TimeStamp")

        return df

    def _get_categorical_normalization_exprs(self) -> list[pl.Expr]:
        """Expressions for normalizing categorical fields"""
        categorical_columns = ["Shift", "TruckFleet", "Status", "Category", "Event"]
        return [
            pl.when(
                pl.col(col).is_null() | (pl.col(col) == "") | (pl.col(col) == "NaN")
            )
            .then(pl.lit("NaN"))
            .otherwise(
                pl.col(col).str.strip().str.to_uppercase()
            )  # Clean and standardize
            .alias(col)
            for col in categorical_columns
        ]

    def _get_duration_validation_exprs(self) -> list[pl.Expr]:
        """Expressions for validating record duration"""
        return [
            # Fix negative durations
            pl.when(pl.col("RecordDuration") < 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("RecordDuration"))
            .alias("RecordDuration"),
            # Cap extremely long durations (more than 8 hours seems unrealistic for single event)
            pl.when(pl.col("RecordDuration") > 480)  # 480 minutes = 8 hours
            .then(pl.lit(480.0))
            .otherwise(pl.col("RecordDuration"))
            .alias("RecordDuration"),
        ]

    def _get_status_normalization_exprs(self) -> list[pl.Expr]:
        """Expressions for standardizing status and event combinations"""
        return [
            # Standardize common status values
            pl.when(pl.col("Status").str.contains("(?i)working|operating|active"))
            .then(pl.lit("WORKING"))
            .when(pl.col("Status").str.contains("(?i)idle|waiting|standby"))
            .then(pl.lit("IDLE"))
            .when(pl.col("Status").str.contains("(?i)maintenance|repair|service"))
            .then(pl.lit("MAINTENANCE"))
            .when(pl.col("Status").str.contains("(?i)moving|traveling|transit"))
            .then(pl.lit("MOVING"))
            .when(pl.col("Status").str.contains("(?i)loading|dumping|spotting"))
            .then(pl.lit("LOADING_DUMPING"))
            .otherwise(pl.col("Status"))
            .alias("Status"),
            # Standardize category values
            pl.when(pl.col("Category").str.contains("(?i)productive|production"))
            .then(pl.lit("PRODUCTIVE"))
            .when(pl.col("Category").str.contains("(?i)delay|downtime|non.productive"))
            .then(pl.lit("NON_PRODUCTIVE"))
            .when(pl.col("Category").str.contains("(?i)maintenance|scheduled"))
            .then(pl.lit("MAINTENANCE"))
            .otherwise(pl.col("Category"))
            .alias("Category"),
        ]

    def _count_categorical_fixes(self, df: pl.DataFrame) -> None:
        """Count fixed empty values in categorical fields"""
        categorical_columns = ["Shift", "TruckFleet", "Status", "Category", "Event"]
        for col in categorical_columns:
            if col in df.columns:
                null_count = df.filter(
                    pl.col(col).is_null() | (pl.col(col) == "") | (pl.col(col) == "NaN")
                ).height
                self.metrics["categorical_empty_fixed"] += null_count

    def _count_duration_fixes(self, df: pl.DataFrame) -> None:
        """Count negative duration values that were fixed"""
        if "RecordDuration" in df.columns:
            negative_count = df.filter(pl.col("RecordDuration") < 0).height
            self.metrics["negative_durations_fixed"] += negative_count

    def _remove_duplicate_events(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate events for same equipment at same timestamp"""
        before_duplicates = df.height

        # Remove exact duplicates first
        df = df.unique()

        # Remove duplicates based on Equipment + TimeStamp, keeping the one with longest duration
        df = df.with_row_count("row_id").sort("RecordDuration", descending=True)
        df = df.unique(subset=["Equipment", "TimeStamp"], keep="first")
        df = df.drop("row_id")

        self.metrics["duplicate_events_removed"] = before_duplicates - df.height
        return df

    def _validate_timestamp_sequences(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate timestamp sequences per equipment"""
        before_validation = df.height

        # Add a flag for timestamp sequence validation
        df = df.with_columns(
            [
                # Check if timestamps are in logical sequence per equipment
                pl.col("TimeStamp")
                .sort()
                .over("Equipment")
                .alias("expected_timestamp"),
                pl.col("TimeStamp").alias("actual_timestamp"),
            ]
        )

        # For now, just count potential issues but don't filter
        # In a real scenario, you might want to implement more sophisticated validation
        timestamp_issues = df.filter(
            pl.col("actual_timestamp") != pl.col("expected_timestamp")
        ).height

        self.metrics["timestamp_sequence_errors"] = timestamp_issues

        # Clean up temporary columns
        df = df.drop(["expected_timestamp", "actual_timestamp"])

        return df

    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply filters for data quality"""

        # Filter out records with invalid status-category combinations
        before_status_filter = df.height

        # Define valid combinations (adjust based on your business rules)
        invalid_combinations = (
            # WORKING status should not have NON_PRODUCTIVE category
            ((pl.col("Status") == "WORKING") & (pl.col("Category") == "NON_PRODUCTIVE"))
            |
            # MAINTENANCE status should have MAINTENANCE category
            (
                (pl.col("Status") == "MAINTENANCE")
                & (pl.col("Category") != "MAINTENANCE")
            )
            |
            # Zero duration records are usually not meaningful for time modeling
            (pl.col("RecordDuration") == 0)
        )

        df = df.filter(~invalid_combinations)
        self.metrics["invalid_status_combinations"] = before_status_filter - df.height

        return df

    def _calculate_time_model_fields(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate derived fields for time model analysis"""
        return df.with_columns(
            [
                # Calculate cumulative time per equipment per shift
                pl.col("RecordDuration")
                .cumsum()
                .over(["Equipment", "ShiftDate"])
                .alias("CumulativeTime"),
                # Calculate time since last event per equipment
                (
                    pl.col("TimeStamp")
                    - pl.col("TimeStamp").shift(1).over(["Equipment", "ShiftDate"])
                )
                .dt.total_minutes()
                .alias("TimeSinceLastEvent"),
                # Calculate percentage of shift time
                (
                    pl.col("RecordDuration")
                    / pl.col("RecordDuration").sum().over(["Equipment", "ShiftDate"])
                    * 100
                ).alias("ShiftTimePercentage"),
                # Add shift hour for time pattern analysis
                pl.col("TimeStamp").dt.hour().alias("ShiftHour"),
                # Add day of week for pattern analysis
                pl.col("TimeStamp").dt.weekday().alias("DayOfWeek"),
                # Create status transition flag
                (
                    pl.col("Status")
                    != pl.col("Status").shift(1).over(["Equipment", "ShiftDate"])
                ).alias("StatusTransition"),
                # Calculate productivity score (simple example)
                pl.when(pl.col("Category") == "PRODUCTIVE")
                .then(pl.col("RecordDuration"))
                .otherwise(pl.lit(0.0))
                .alias("ProductiveTime"),
                # Calculate delay time
                pl.when(pl.col("Category") == "NON_PRODUCTIVE")
                .then(pl.col("RecordDuration"))
                .otherwise(pl.lit(0.0))
                .alias("DelayTime"),
                # Add equipment utilization flag
                pl.when(
                    pl.col("Status").is_in(["WORKING", "LOADING_DUMPING", "MOVING"])
                )
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
                .alias("IsUtilized"),
            ]
        )
