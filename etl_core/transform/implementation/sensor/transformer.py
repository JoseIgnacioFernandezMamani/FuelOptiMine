from typing import List, Type, Optional
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.sensor_schemas import SensorSchema
from etl_core.transform.utils.unit_converter import (
    get_coordinate_conversion_exprs,
    get_geo_validation_expr,
)
from pydantic import BaseModel
import polars as pl

from etl_core.transform.utils.data_normalizer import (
    get_categorical_normalization_exprs,
    count_null_empty_categorical_values,
)


class SensorTransformer(BaseTransformer):
    """Optimized transformer for sensor data using Polars expressions"""

    def __init__(self):
        super().__init__()
        self.metrics.update(
            {
                "outliers_removed": 0,
                "invalid_geo_records": 0,
                "categorical_null_empty_replaced": 0,
            }
        )

        # Define categorical columns as class attribute
        self.categorical_columns = [
            "Shift",
            "FuelGauge",
            "Ralenti",
            "TruckFleet",
        ]

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
        categorical_exprs = get_categorical_normalization_exprs(
            self.categorical_columns
        )
        outlier_exprs = self._get_outlier_handling_exprs()
        geo_validation_expr = get_geo_validation_expr()

        # 2. Apply all transformations in a single step
        df = df.with_columns(conversion_exprs + categorical_exprs + outlier_exprs)

        # 3. Count fixed categorical values using external function
        self.metrics["categorical_null_empty_replaced"] = (
            count_null_empty_categorical_values(df, self.categorical_columns)
        )

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

    def _get_outlier_handling_exprs(self) -> list[pl.Expr]:
        """Expressions for handling outlier values (convert to null)"""
        return [
            pl.when(
                pl.col("FuelLevelLiters") > 4500
            )  # Based on the documentation and the constant equipment values, this is the maximum valid value.
            .then(None)
            .otherwise(pl.col("FuelLevelLiters"))
            .alias("FuelLevelLiters"),
            pl.when(
                pl.col("Speed") > 60
            )  # Given the documentation, the maximum speed limit is 60.
            .then(None)
            .otherwise(pl.col("Speed"))
            .alias("Speed"),
        ]

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
