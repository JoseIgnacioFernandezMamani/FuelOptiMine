from typing import List, Type, Optional
from etl_core.transform.core import BaseTransformer
from etl_core.utils.fuel_supply_schemas import FuelSupplySchema
from etl_core.transform.utils import (
    get_categorical_normalization_exprs,
    count_null_empty_categorical_values,
)

from pydantic import BaseModel
import polars as pl
from polars import Expr
from etl_core.utils import TRUCK_SPECS


class FuelSupplyTransformer(BaseTransformer):
    """Optimized transformer for fuel supply data using Polars expressions"""

    def __init__(self) -> None:
        super().__init__()
        # Initialize domain-specific metrics
        self.metrics.update(
            {
                "invalid_truck_models": 0,
                "invalid_origin_records": 0,
                "outliers_removed": 0,
                "categorical_null_empty_replaced": 0,
            }
        )
        self.categorical_columns = ["TruckFleet", "Shift"]
        self.VALID_ORIGINS = {"P068", "SST", "SURTIDOR-TRUCKSHOP"}
        self.VALID_TRUCK_FLEETS = {"CAT789C", "CAT793D"}

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

    def transform(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        if df.is_empty() or "TruckFleet" not in df.columns:
            print("⚠️ DataFrame vacío o falta columna TruckFleet")
            return df

        # 1. Aplicar transformaciones: normalización categórica, outliers, validaciones
        categorical_exprs: list[Expr] = get_categorical_normalization_exprs(
            self.categorical_columns, default_value="NoData"
        )
        outlier_exprs: list[Expr] = self._get_outlier_handling_exprs()
        domain_validation_exprs: list[Expr] = self._get_domain_validation_exprs()

        df = df.with_columns(
            categorical_exprs + outlier_exprs + domain_validation_exprs
        )

        # 2. Contar valores categóricos corregidos
        self.metrics["categorical_null_empty_replaced"] = (
            count_null_empty_categorical_values(
                df, self.categorical_columns, default_value="NoData"
            )
        )

        # 3. Filtrar por validaciones de dominio
        df = self._apply_domain_filters(df)

        # 4. Actualizar métricas finales
        self.metrics["after_transform_records"] = df.height
        if self.metrics["initial_records"] > 0:
            self.metrics["final_data_percentage"] = round(
                (df.height / self.metrics["initial_records"]) * 100, 2
            )
        return df.sort(["ShiftDate", "TimeStamp"])

    def _get_outlier_handling_exprs(self) -> List[pl.Expr]:
        """Convertir valores fuera de rango a nulos"""
        return [
            pl.when(pl.col("FuelLevelLiters") > 4500)
            .then(None)
            .otherwise(pl.col("FuelLevelLiters"))
            .alias("FuelLevelLiters"),
            pl.when((pl.col("FuelLevel") < 0) | (pl.col("FuelLevel") > 100))
            .then(None)
            .otherwise(pl.col("FuelLevel"))
            .alias("FuelLevel"),
        ]

    def _get_domain_validation_exprs(self) -> List[pl.Expr]:
        """Validaciones específicas del dominio"""
        return [
            pl.col("TruckFleet")
            .str.to_uppercase()
            .is_in(self.VALID_TRUCK_FLEETS)
            .alias("__valid_model"),
            pl.col("Origin")
            .str.to_uppercase()
            .is_in(self.VALID_ORIGINS)
            .alias("__valid_origin"),
        ]

    def _apply_domain_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filtrar registros inválidos y actualizar métricas"""
        invalid_model_count: int = df.filter(~pl.col("__valid_model")).height
        invalid_origin_count: int = df.filter(~pl.col("__valid_origin")).height
        outlier_count: int = df.filter(
            pl.col("FuelLevelLiters").is_null() | pl.col("FuelLevel").is_null()
        ).height

        df = df.filter(
            pl.col("__valid_model")
            & pl.col("__valid_origin")
            & pl.col("FuelLevelLiters").is_not_null()
            & pl.col("FuelLevel").is_not_null()
        ).drop(["__valid_model", "__valid_origin"])

        # convert model to standar
        df = df.with_columns(
            pl.col("TruckFleet")
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"CAT\s*789C", "CAT 789C")
            .str.replace_all(r"CAT\s*793D", "CAT 793D")
            .alias("TruckFleet")
        )

        self.metrics["invalid_truck_models"] = invalid_model_count
        self.metrics["invalid_origin_records"] = invalid_origin_count
        self.metrics["outliers_removed"] = outlier_count

        return df
