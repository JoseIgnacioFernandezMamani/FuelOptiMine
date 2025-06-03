from typing import List, Type
from etl_core.transform.core.base_transformer import BaseTransformer
from etl_core.utils.cycle_schemas import CycleSchema
from pydantic import BaseModel
import polars as pl
from datetime import datetime


class CycleTransformer(BaseTransformer):
    """Transformador para datos de ciclos de carga y transporte minero"""

    def __init__(self):
        super().__init__()
        self._tonnage_threshold = 0.1  # 10% diferencia tolerada

    @property
    def mandatory_columns(self) -> List[str]:
        return ["ShiftDate", "Equipment"]

    @property
    def schema_model(self) -> Type[BaseModel]:
        return CycleSchema

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ejecuta el pipeline de transformación de ciclos"""
        try:
            if df.is_empty():
                return df

            # Validación y preparación inicial
            processed_df = self._preprocess_data(df)

            # Etapa 1: Cálculos básicos de tiempo y distancia
            stage1 = processed_df.with_columns(
                [
                    self._calculate_phase_duration("TravelingEmpty", "E"),
                    self._calculate_phase_duration("LoadingMaterial", "E"),
                    self._calculate_phase_duration("Hauling", "L"),
                    self._calculate_phase_duration("UnloadingMaterial", "L"),
                    self._calculate_total_empty_distance(),
                    self._calculate_total_loaded_distance(),
                ]
            )

            # Etapa 2: Métricas de eficiencia
            stage2 = stage1.with_columns(
                [
                    self._calculate_payload_efficiency(),
                    self._calculate_cycle_time(),
                    self._calculate_speed("Empty"),
                    self._calculate_speed("Loaded"),
                ]
            )

            # Etapa 3: Clasificaciones operacionales
            return stage2.with_columns(
                [self._classify_cycle_efficiency(), self._detect_operational_issues()]
            )

        except Exception as e:
            raise RuntimeError(f"Error en transformación de ciclos: {str(e)}")

    def _preprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocesamiento inicial de datos"""
        return df.sort(["Equipment", "TimeStamp"]).with_columns(
            [
                pl.col("MeasuredTonnage").fill_nan(0),
                pl.col("ReportedTonnage").fill_nan(0),
            ]
        )

    def _calculate_phase_duration(self, phase: str, load_status: str) -> pl.Expr:
        """Calcula la duración de una fase del ciclo"""
        start_col = f"{load_status}_{phase}Start"
        end_col = f"{load_status}_{phase}End"
        return (
            (pl.col(end_col).cast(pl.Datetime) - pl.col(start_col).cast(pl.Datetime))
            .dt.total_seconds()
            .alias(f"{phase}Duration")
        )

    def _calculate_total_empty_distance(self) -> pl.Expr:
        """Calcula distancia total recorrida vacío"""
        return (pl.col("DistanceEmpty") + pl.col("EquivalentDistance")).alias(
            "TotalEmptyDistance"
        )

    def _calculate_total_loaded_distance(self) -> pl.Expr:
        """Calcula distancia total recorrida cargado"""
        return (pl.col("DistanceLoaded") + pl.col("EquivalentDistance")).alias(
            "TotalLoadedDistance"
        )

    def _calculate_payload_efficiency(self) -> pl.Expr:
        """Eficiencia de carga: Medido vs Reportado"""
        return (
            (
                pl.col("MeasuredTonnage")
                / pl.col("ReportedTonnage").clip(lower_bound=0.1)
            )
            .fill_nan(0)
            .alias("PayloadEfficiency")
        )

    def _calculate_cycle_time(self) -> pl.Expr:
        """Tiempo total del ciclo completo"""
        return (
            pl.col("TravelingEmptyDuration")
            + pl.col("LoadingMaterialDuration")
            + pl.col("HaulingDuration")
            + pl.col("UnloadingMaterialDuration")
        ).alias("TotalCycleTime")

    def _calculate_speed(self, load_status: str) -> pl.Expr:
        """Calcula velocidad promedio por estado de carga"""
        distance = pl.col(f"Total{load_status}Distance")
        duration = pl.col(f"{load_status}TravelDuration")
        return (
            (distance / (duration / 3600).clip(lower_bound=0.1))
            .fill_nan(0)
            .alias(f"{load_status}Speed")
        )

    def _classify_cycle_efficiency(self) -> pl.Expr:
        """Clasificación de eficiencia del ciclo"""
        return (
            pl.when(pl.col("PayloadEfficiency") >= 0.95)
            .then("Optimal")
            .when(pl.col("PayloadEfficiency") >= 0.85)
            .then("Acceptable")
            .otherwise("Inefficient")
            .alias("EfficiencyClass")
        )

    def _detect_operational_issues(self) -> pl.Expr:
        """Detección de problemas operacionales"""
        return (
            pl.when(pl.col("LoadingMaterialDuration") > 1800)
            .then("Loading Delay")
            .when(pl.col("UnloadingMaterialDuration") > 600)
            .then("Unloading Delay")
            .when(pl.col("EmptySpeed") < 5.0)
            .then("Low Empty Speed")
            .otherwise("Normal")
            .alias("OperationalStatus")
        )
