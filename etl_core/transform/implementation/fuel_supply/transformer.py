from typing import List, Type
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.schemas.fuel_supply_schema import FuelSupplySchema
from pydantic import BaseModel
import polars as pl


class FuelSupplyTransformer(BaseTransformer):
    """Optimized transformer for fuel supply data using Polars expressions"""

    def __init__(self):
        super().__init__()
        # Initialize domain-specific metrics
        self.metrics.update(
            {
                "invalid_truck_models": 0,
                "invalid_origin_records": 0,
                "outliers_removed": 0,
                "categorical_empty_fixed": 0,
            }
        )

    @property
    def mandatory_columns(self) -> List[str]:
        """Dynamically get mandatory columns from Pydantic schema"""
        return [
            field_name
            for field_name, field in FuelSupplySchema.model_fields.items()
            if field.is_required()
        ]

    @property
    def schema_model(self) -> Type[BaseModel]:
        """Pydantic schema for data validation"""
        return FuelSupplySchema

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimized transformation pipeline using expressions"""
        # 1. Collect all transformation expressions
        categorical_exprs = self._get_categorical_normalization_exprs()
        outlier_exprs = self._get_outlier_handling_exprs()
        domain_validation_exprs = self._get_domain_validation_exprs()

        # 2. Apply all transformations in a single step
        df = df.with_columns(
            categorical_exprs + outlier_exprs + domain_validation_exprs
        )

        # 3. Count fixed categorical values
        self._count_categorical_fixes(df)

        # 4. Apply filters and update metrics
        df = self._apply_domain_filters(df)

        # 5. Update final metrics
        self.metrics["after_transform_records"] = df.height
        return df

    def _get_categorical_normalization_exprs(self) -> list[pl.Expr]:
        """Normalize categorical fields (TruckFleet and Shift)"""
        return [
            # Normalize truck model names
            pl.when(pl.col("TruckFleet").is_null() | (pl.col("TruckFleet") == ""))
            .then(pl.lit("UnknownModel"))
            .otherwise(pl.col("TruckFleet"))
            .alias("TruckFleet"),
            # Normalize shift values
            pl.when(pl.col("Shift").is_null() | (pl.col("Shift") == ""))
            .then(pl.lit("UnknownShift"))
            .otherwise(pl.col("Shift"))
            .alias("Shift"),
        ]

    def _get_outlier_handling_exprs(self) -> list[pl.Expr]:
        """Handle outlier values in fuel metrics"""
        return [
            # Handle impossible fuel liters
            pl.when(pl.col("FuelLevelLiters") > 5000)
            .then(None)
            .otherwise(pl.col("FuelLevelLiters"))
            .alias("FuelLevelLiters"),
            # Handle impossible fuel percentages
            pl.when((pl.col("FuelLevel") < 0) | (pl.col("FuelLevel") > 100))
            .then(None)
            .otherwise(pl.col("FuelLevel"))
            .alias("FuelLevel"),
        ]

    def _get_domain_validation_exprs(self) -> list[pl.Expr]:
        """Create validation flags for domain-specific rules"""
        return [
            # Flag for valid truck models
            pl.col("TruckFleet").is_in(TRUCK_SPECS.keys()).alias("__valid_model"),
            # Flag for valid origin format
            pl.col("Origin").str.contains(r"^[PC]\d{3}$").alias("__valid_origin"),
        ]

    def _count_categorical_fixes(self, df: pl.DataFrame) -> None:
        """Count fixed empty values in categorical fields"""
        for col in ["TruckFleet", "Shift"]:
            if col in df.columns:
                # Count records that were fixed (originally null or empty)
                fixed_count = df.filter(pl.col(col).str.contains("Unknown")).height
                self.metrics["categorical_empty_fixed"] += fixed_count

    def _apply_domain_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply domain filters and update metrics using efficient expressions"""
        # Count invalid records before filtering
        invalid_model_count = df.filter(pl.col("__valid_model").not_()).height
        invalid_origin_count = df.filter(pl.col("__valid_origin").not_()).height

        # Count outliers
        outlier_count = df.filter(
            pl.col("FuelLevelLiters").is_null() | pl.col("FuelLevel").is_null()
        ).height

        # Apply all domain filters
        df = df.filter(
            pl.col("__valid_model")
            & pl.col("__valid_origin")
            & pl.col("FuelLevelLiters").is_not_null()
            & pl.col("FuelLevel").is_not_null()
        ).drop(["__valid_model", "__valid_origin"])

        # Update metrics
        self.metrics["invalid_truck_models"] = invalid_model_count
        self.metrics["invalid_origin_records"] = invalid_origin_count
        self.metrics["outliers_removed"] = outlier_count

        return df
