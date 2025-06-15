from typing import List, Type, Optional
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.sensor_schemas import SensorSchema
from etl_core.transform.utils.unit_converter import (
    get_coordinate_conversion_exprs,
    get_geo_validation_expr,
)
from pydantic import BaseModel
import polars as pl


class SensorTransformer(BaseTransformer):
    """Optimized transformer for sensor data using Polars expressions"""

    def __init__(self):
        super().__init__()
        self.metrics.update(
            {
                "outliers_removed": 0,
                "invalid_geo_records": 0,
                "categorical_empty_fixed": 0,
            }
        )

    @property
    def mandatory_columns(self) -> List[str]:
        """Dynamically get mandatory columns from Pydantic schema"""
        return [
            field_name
            for field_name, field in SensorSchema.model_fields.items()
            if field.is_required()
        ]

    @property
    def schema_model(self) -> Type[BaseModel]:
        """Pydantic schema for data validation"""
        return SensorSchema

    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Optimized transformation pipeline using expressions"""

        # 1. Collect all transformation expressions
        conversion_exprs = get_coordinate_conversion_exprs()
        categorical_exprs = self._get_categorical_normalization_exprs()
        outlier_exprs = self._get_outlier_handling_exprs()
        geo_validation_expr = get_geo_validation_expr()

        # 2. Apply all transformations in a single step
        df = df.with_columns(conversion_exprs + categorical_exprs + outlier_exprs)

        # 3. Count fixed categorical values
        self._count_categorical_fixes(df)

        # 4. Apply filters and update metrics
        df = self._apply_filters(df, geo_validation_expr)

        # 5. Update final metrics
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )

        # finally, sort by ShiftDate and TimeStamp
        df = df.sort("ShiftDate", "TimeStamp")

        return df

    def _get_categorical_normalization_exprs(self) -> list[pl.Expr]:
        """Expressions for normalizing categorical fields"""
        categorical_columns = ["Shift", "FuelGauge", "Ralenti", "TruckFleet"]
        return [
            pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
            .then(pl.lit("NoData"))
            .otherwise(pl.col(col))
            .alias(col)
            for col in categorical_columns
        ]

    def _get_outlier_handling_exprs(self) -> list[pl.Expr]:
        """Expressions for handling outlier values (convert to null)"""
        return [
            pl.when(pl.col("FuelLevelLiters") > 5000)
            .then(None)
            .otherwise(pl.col("FuelLevelLiters"))
            .alias("FuelLevelLiters"),
            pl.when(pl.col("Speed") > 150)
            .then(None)
            .otherwise(pl.col("Speed"))
            .alias("Speed"),
        ]

    def _count_categorical_fixes(self, df: pl.DataFrame) -> None:
        """Count fixed empty values in categorical fields"""
        for col in ["Shift", "FuelGauge", "Ralenti", "TruckFleet"]:
            if col in df.columns:
                null_count = df.filter(
                    pl.col(col).is_null() | (pl.col(col) == "")
                ).height
                self.metrics["categorical_empty_fixed"] += null_count

    def _apply_filters(
        self, df: pl.DataFrame, geo_validation_expr: pl.Expr
    ) -> pl.DataFrame:
        """Apply all filters and update metrics"""
        # Filter outliers (values converted to null)
        before_outliers = df.height
        df = df.filter(
            pl.col("FuelLevelLiters").is_not_null() & pl.col("Speed").is_not_null()
        )
        self.metrics["outliers_removed"] = before_outliers - df.height

        # Filter invalid geo records
        before_geo = df.height
        df = df.filter(geo_validation_expr)
        self.metrics["invalid_geo_records"] = before_geo - df.height

        return df
