from typing import List, Type
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.equipment_constants import TRUCK_SPECS
from etl_core.utils.fuel_supply_schemas import FuelSupplySchema
from pydantic import BaseModel
import polars as pl


class FuelSupplyTransformer(BaseTransformer):
    """Transformador para datos de abastecimiento de combustible"""

    def __init__(self):
        super().__init__()

    @property
    def mandatory_columns(self) -> List[str]:
        return [
            field_name
            for field_name, field in FuelSupplySchema.model_fields.items()
            if field.is_required()
        ]

    @property
    def schema_model(self) -> Type[BaseModel]:
        return FuelSupplySchema

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ejecuta el pipeline de transformación de despachos"""
        try:
            if df.is_empty():
                return df

            # Preprocesamiento básico
            processed_df = df.sort(["Veh", "fin_desp"])

            # Etapa 1: Cálculos fundamentales
            stage1 = processed_df.with_columns(*self._standardize_columns())

            stage2 = stage1.with_columns(
                self._calculate_time_between_refuels(),
                self._shift(),
                self._fuel_level(),
            )

            stage2 = stage2.drop(["Veh", "Descripcion", "fin_desp", "volumCorregido"])

            return stage2

        except Exception as e:
            raise RuntimeError(f"Error en transformación de despachos: {str(e)}")

    def _standardize_columns(self) -> list[pl.Expr]:
        """Estandariza nombres de columnas a minúsculas y sin espacios"""

        # Transformaciones básicas
        return [
            pl.col("fin_desp").dt.date().alias("ShiftDate"),
            pl.col("fin_desp").alias("TimeStamp"),
            pl.col("Veh")
            .str.slice(0, 3)
            .map_elements(lambda x: f"T-{x}", return_dtype=pl.Utf8)
            .alias("Equipment"),
            pl.col("Descripcion").str.replace_all(r"[\s-]+", "").alias("TruckFleet"),
            pl.col("volumCorregido").alias("FuelLevelLiters"),
        ]

    def _calculate_time_between_refuels(self) -> pl.Expr:
        """Tiempo desde el último reabastecimiento (segundos)"""
        return (
            (pl.col("TimeStamp") - pl.col("TimeStamp").shift(1).over("Equipment"))
            .dt.total_seconds()
            .alias("LastRefuel")
        )

    def _fuel_level(self) -> pl.Expr:
        """Nivel de combustible llenado"""
        truck_capacity_map = {k: v["capacity"] for k, v in TRUCK_SPECS.items()}

        truck_capacity = pl.col("Equipment").replace(truck_capacity_map, default=3000)

        return (
            (pl.col("FuelLevelLiters") / truck_capacity * 100)
            .clip(upper_bound=100.0)
            .round(4)
            .alias("FuelLevel")
            .cast(pl.Float64)
        )

    def _shift(self) -> pl.Expr:
        """Clasifica el turno como Diurno (D) o Nocturno (N)"""
        hour = pl.col("TimeStamp").dt.hour()
        return (
            pl.when((hour >= 7) & (hour <= 19))
            .then(pl.lit("D"))
            .otherwise(pl.lit("N"))
            .alias("Shift")  # Nombre único para la nueva columna
        )
