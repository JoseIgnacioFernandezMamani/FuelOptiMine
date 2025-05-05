from typing import List, Type
from transform.core.base_transformer import BaseTransformer
from .schemas import FuelSupplySchema
from pydantic import BaseModel
import polars as pl
from datetime import datetime

class FuelSupplyTransformer(BaseTransformer):
    """Transformador para datos de despacho de mineral"""
    
    def __init__(self):
        super().__init__()
        self._volume_tolerance = 0.05  # 5% de tolerancia

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
            stage1 = processed_df.with_columns([
                self._calculate_FuelSupply_duration(),
                self._convert_volume_units(),
                self._calculate_theoretical_volume(),
                self._detect_priority_material()
            ])
            
            # Etapa 2: Métricas de eficiencia
            stage2 = stage1.with_columns([
                self._calculate_volume_efficiency(),
                self._calculate_FuelSupply_yield(),
                self._classify_FuelSupply_quality()
            ])
            
            # Etapa 3: Análisis operacional
            return stage2.with_columns([
                self._identify_loading_anomalies(),
                self._calculate_FuelSupply_frequency()
            ])

        except Exception as e:
            raise RuntimeError(f"Error en transformación de despachos: {str(e)}")


    def _calculate_FuelSupply_duration(self) -> pl.Expr:
        """Calcula duración desde inicio de carga hasta despacho"""
        return (
            pl.col("fin_desp") - 
            pl.col("inicio_carga").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        ).dt.total_minutes().alias("duracion_carga")

    def _convert_volume_units(self) -> pl.Expr:
        """Convierte volumen a toneladas métricas (asumiendo m³ -> ton)"""
        density_factor = 1.8  # Factor de densidad promedio (ton/m³)
        return (pl.col("volumCorregido") * density_factor).alias("tonelaje_real")

    def _calculate_theoretical_volume(self) -> pl.Expr:
        """Calcula volumen teórico basado capacidad del vehículo"""
        capacity_map = {
            "CAM-789": 180.0,
            "CAT-777": 160.0,
            "KOM-930": 220.0
        }
        return pl.col("Veh").map_dict(capacity_map).alias("tonelaje_teorico")

    def _detect_priority_material(self) -> pl.Expr:
        """Identifica despachos de alta prioridad"""
        return (
            pl.col("Descripcion").str.contains("urgente|prioritario|crítico")
            .fill_null(False)
            .alias("prioridad_alta")
        )

    def _calculate_volume_efficiency(self) -> pl.Expr:
        """Eficiencia de carga: Real vs Teórico"""
        return (
            pl.col("tonelaje_real") / 
            pl.col("tonelaje_teorico").clip(lower_bound=0.1)
        ).alias("eficiencia_carga")

    def _calculate_FuelSupply_yield(self) -> pl.Expr:
        """Rendimiento por hora de operación"""
        return (
            pl.col("tonelaje_real") / 
            (pl.col("duracion_carga") / 60).clip(lower_bound=0.1)
        ).alias("rendimiento_ton_hora")

    def _classify_FuelSupply_quality(self) -> pl.Expr:
        """Clasificación de calidad del despacho"""
        return (
            pl.when(pl.col("eficiencia_carga") >= 0.95)
            .then("Óptimo")
            .when(pl.col("eficiencia_carga") >= 0.85)
            .then("Aceptable")
            .otherwise("Revisión")
            .alias("clasificacion_calidad")
        )

    def _identify_loading_anomalies(self) -> pl.Expr:
        """Detección de anomalías en carga"""
        return (
            pl.when(
                (pl.col("tonelaje_real") < pl.col("tonelaje_teorico") * 0.85) |
                (pl.col("tonelaje_real") > pl.col("tonelaje_teorico") * 1.10)
            )
            .then(True)
            .otherwise(False)
            .alias("anomalia_carga")
        )

    def _calculate_FuelSupply_frequency(self) -> pl.Expr:
        """Calcula frecuencia de despachos por vehículo"""
        return (
            pl.col("Veh").cumcount().over("Veh")
            .alias("conteo_despachos")
        )