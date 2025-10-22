import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import json
from etl_core.load.utils import create_client, CH_CONFIG


class CycleDataEDA:
    """
    Análisis exploratorio de datos de ciclos mineros.
    Enfocado en variables categóricas y análisis temporal.
    """

    def __init__(self, truck_id: str = "T-210") -> None:
        self.cycle_df: pl.DataFrame = pl.DataFrame()
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._data_loaded: bool = False
        self._stats_generated: bool = False
        self.truck_id = truck_id
        self.client = create_client(CH_CONFIG)

        # Definición de columnas
        self.categorical_cols = [
            "Shovel",
            "ShovelModel",
            "StageType",
            "LoadingZone",
            "Material",
            "DestinationType",
            "Destination",
        ]

        self.numeric_cols = [
            "StageSequence",
            "MeasuredTonnage",
            "ReportedTonnage",
            "Distance",
            "TimeEfficiencyPercentage",
        ]

        self.datetime_cols = ["TimeStampIni", "TimeStampFin"]

    def get_dataframe(self) -> pl.DataFrame:
        """Obtener DataFrame cargado"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")
        return self.cycle_df

    def _load_cycle_data(self):
        """Cargar datos de ciclos desde ClickHouse"""
        try:
            query = f"""
            SELECT 
                CycleId,
                Shovel,
                ShovelModel,
                StageType,
                StageSequence,
                TimeStampIni,
                TimeStampFin,
                LoadingZone,
                Material,
                MeasuredTonnage,
                ReportedTonnage,
                DestinationType,
                Destination,
                Distance,
                TimeEfficiencyPercentage
            FROM xgboost_fuel
            WHERE Equipment = '{self.truck_id}' 
            AND CycleId IS NOT NULL 
            ORDER BY TimeStampIni
            """

            pandas_df = self.client.query_df(query)

            if pandas_df.empty:
                raise RuntimeError(
                    f"No se encontraron datos para el equipo {self.truck_id}"
                )

            self.cycle_df = pl.from_pandas(pandas_df)

            # Ordenar por TimeStamp
            self.cycle_df = self.cycle_df.sort("TimeStampIni")

            # PASO 1: Calcular columnas base y temporales
            self.cycle_df = self.cycle_df.with_columns(
                [
                    # Duración del ciclo en segundos
                    (pl.col("TimeStampIni") - pl.col("TimeStampFin"))
                    .dt.total_seconds()
                    .alias("CycleDurationSeconds"),
                    # Diferencia entre tonelaje medido y reportado
                    (pl.col("MeasuredTonnage") - pl.col("ReportedTonnage")).alias(
                        "TonnageDifference"
                    ),
                    # Extraer componentes temporales
                    pl.col("TimeStampIni").dt.hour().alias("Hour"),
                    pl.col("TimeStampIni").dt.weekday().alias("Weekday"),
                    pl.col("TimeStampIni").dt.month().alias("Month"),
                    # Turno (mañana: 7-15, tarde: 15-23, noche: 23-7)
                    pl.when(
                        (pl.col("TimeStampIni").dt.hour() >= 7)
                        & (pl.col("TimeStampIni").dt.hour() < 19)
                    )
                    .then(pl.lit("D"))
                    .otherwise(pl.lit("N"))
                    .alias("Shift"),
                ]
            )

            # Actualizar listas de columnas
            self.numeric_cols.extend(
                [
                    "CycleDurationSeconds",
                    "TonnageDifference",
                ]
            )
            self.categorical_cols.append("Shift")

            self._data_loaded = True
            self._stats_cache = {}
            self._stats_generated = False

        except Exception as e:
            raise RuntimeError(f"Error al cargar datos de ciclos: {str(e)}")

    def _generate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Generar estadísticas detalladas para columnas categóricas, numéricas y temporales
        aplicando los filtros adecuados según StageSequence y casos especiales.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        df = self.cycle_df
        stats: Dict[str, Dict[str, Any]] = {}

        # === Casos especiales ===
        if "CycleId" in df.columns:
            stats["CycleId"] = {
                "type": "cycle_count",
                "total_cycles": df["CycleId"].n_unique(),
            }

        if "StageSequence" in df.columns:
            stats["StageSequence"] = {
                "type": "stage_sequence",
                "unique_stages": df["StageSequence"].unique().to_list(),
            }

        if "StageType" in df.columns:
            stats["StageType"] = {
                "type": "stage_types",
                "unique_stage_types": df["StageType"].unique().to_list(),
            }

        # === Definir filtros ===
        df_stage4 = df.filter(pl.col("StageSequence") == 4)
        df_stage8 = df.filter(pl.col("StageSequence") == 8)
        df_stage4_8 = df.filter(pl.col("StageSequence").is_in([4, 8]))

        # === 1. CATEGÓRICAS ===
        # Stage 4 -> Shovel, ShovelModel, LoadingZone
        for col in ["Shovel", "ShovelModel", "LoadingZone"]:
            if col in df.columns:
                vc = df_stage4[col].value_counts().sort("count", descending=True)
                stats[col] = {
                    "type": "categorical",
                    "stage_filter": "StageSequence == 4",
                    "total_unique_values": df_stage4[col].n_unique(),
                    "null_count": df_stage4[col].null_count(),
                    "values": vc.to_dicts(),
                    "total_records_analyzed": len(df_stage4),
                }

        # Stage 8 -> Material, DestinationType, Destination
        for col in ["Material", "DestinationType", "Destination"]:
            if col in df.columns:
                vc = df_stage8[col].value_counts().sort("count", descending=True)
                stats[col] = {
                    "type": "categorical",
                    "stage_filter": "StageSequence == 8",
                    "total_unique_values": df_stage8[col].n_unique(),
                    "null_count": df_stage8[col].null_count(),
                    "values": vc.to_dicts(),
                    "total_records_analyzed": len(df_stage8),
                }

        # 2.2 MeasuredTonnage & ReportedTonnage (solo StageSequence == 8)
        for col in ["MeasuredTonnage", "ReportedTonnage"]:
            if col in df.columns:
                non_null = df_stage8[col]
                stats[col] = {
                    "type": "numeric",
                    "stage_filter": "StageSequence == 8",
                    "mean": non_null.mean(),
                    "median": non_null.median(),
                    "min": non_null.min(),
                    "max": non_null.max(),
                    "std_dev": non_null.std(),
                    "q1": non_null.quantile(0.25),
                    "q3": non_null.quantile(0.75),
                    "p5": non_null.quantile(0.05),
                    "p95": non_null.quantile(0.95),
                    "skewness": non_null.skew(),
                    "kurtosis": non_null.kurtosis(),
                    "cv": (
                        (non_null.std() / non_null.mean()) if non_null.mean() else None
                    ),
                    "non_null_count": len(non_null),
                    "null_count": df_stage8[col].null_count(),
                }

        # 2.3 Distance (StageSequence == 4 o 8)
        if "Distance" in df.columns:
            non_null = df_stage4_8["Distance"]
            stats["Distance"] = {
                "type": "numeric",
                "stage_filter": "StageSequence == 4 OR 8",
                "mean": non_null.mean(),
                "median": non_null.median(),
                "min": non_null.min(),
                "max": non_null.max(),
                "std_dev": non_null.std(),
                "q1": non_null.quantile(0.25),
                "q3": non_null.quantile(0.75),
                "p5": non_null.quantile(0.05),
                "p95": non_null.quantile(0.95),
                "skewness": non_null.skew(),
                "kurtosis": non_null.kurtosis(),
                "cv": (non_null.std() / non_null.mean()) if non_null.mean() else None,
                "non_null_count": len(non_null),
                "null_count": df_stage4_8["Distance"].null_count(),
            }

        # 2.4 TimeEfficiencyPercentage (sin filtro)
        if "TimeEfficiencyPercentage" in df.columns:
            non_null = df["TimeEfficiencyPercentage"]
            stats["TimeEfficiencyPercentage"] = {
                "type": "numeric",
                "stage_filter": "sin filtro",
                "mean": non_null.mean(),
                "median": non_null.median(),
                "min": non_null.min(),
                "max": non_null.max(),
                "std_dev": non_null.std(),
                "q1": non_null.quantile(0.25),
                "q3": non_null.quantile(0.75),
                "p5": non_null.quantile(0.05),
                "p95": non_null.quantile(0.95),
                "skewness": non_null.skew(),
                "kurtosis": non_null.kurtosis(),
                "cv": (non_null.std() / non_null.mean()) if non_null.mean() else None,
                "non_null_count": len(non_null),
                "null_count": df["TimeEfficiencyPercentage"].null_count(),
            }

        # === 3. TEMPORALES ===
        for col in ["TimeStampIni", "TimeStampFin"]:
            if col in df.columns:
                min_date = df[col].min()
                max_date = df[col].max()
                stats[col] = {
                    "type": "datetime",
                    "first_record": (
                        min_date.strftime("%Y-%m-%d %H:%M:%S") if min_date else None
                    ),
                    "last_record": (
                        max_date.strftime("%Y-%m-%d %H:%M:%S") if max_date else None
                    ),
                    "total_duration_days": (
                        ((max_date - min_date).total_seconds() / 86400)
                        if min_date and max_date
                        else None
                    ),
                    "records_per_day": (
                        df.group_by("Date").count().shape[0]
                        if "Date" in df.columns
                        else None
                    ),
                }

        # Análisis especial de componentes temporales
        if "Hour" in df.columns:
            stats["Hour_distribution"] = sorted(
                df["Hour"].value_counts().to_dicts(),
                key=lambda x: x["Hour"],  # 👈 aquí ordenamos por la clave Hour
            )

        if "Weekday" in df.columns:
            stats["Weekday_distribution"] = sorted(
                df["Weekday"].value_counts().to_dicts(), key=lambda x: x["Weekday"]
            )

        if "Month" in df.columns:
            stats["Monthly_distribution"] = sorted(
                df["Month"].value_counts().to_dicts(), key=lambda x: x["Month"]
            )

        self._stats_generated = True
        self._stats_cache = stats
        return stats

    def analyze_time_efficiency(self) -> dict:
        df = self.cycle_df
        stats = {}

        # ──────────────────────────────────────────────
        # 1️⃣ Eficiencia media por StageSequence (1..8)
        # ──────────────────────────────────────────────
        stats["efficiency_by_stage"] = (
            df.group_by("StageSequence")
            .agg(pl.col("TimeEfficiencyPercentage").mean().alias("AvgEfficiency"))
            .sort("StageSequence")
            .to_dicts()
        )

        # ──────────────────────────────────────────────
        # 2️⃣ Crear columna OperationGroup (Empty o Loaded)
        # ──────────────────────────────────────────────
        df = df.with_columns(
            pl.when(pl.col("StageSequence").is_in([1, 2, 3, 4]))
            .then(pl.lit("Empty"))
            .otherwise(pl.lit("Loaded"))
            .alias("OperationGroup")
        )

        # ──────────────────────────────────────────────
        # 3️⃣ Agrupar por ciclo y operación: obtener valores finales y suma de eficiencia
        # ──────────────────────────────────────────────
        grouped = df.group_by(["CycleId", "OperationGroup"]).agg(
            [
                pl.col("Shovel").last(),
                pl.col("ShovelModel").last(),
                pl.col("LoadingZone").last(),
                pl.col("Material").last(),
                pl.col("MeasuredTonnage").last(),
                pl.col("ReportedTonnage").last(),
                pl.col("DestinationType").last(),
                pl.col("Destination").last(),
                pl.col("Distance").last(),
                pl.col("StageSequence").last().alias("StageSequence"),
                pl.col("TimeEfficiencyPercentage").sum().alias("TotalEfficiency"),
            ]
        )
        stats["cycle_group_summary"] = grouped.to_dicts()

        # ──────────────────────────────────────────────
        # 4️⃣ Eficiencia media por factores en etapa 4 y etapa 8 (agrupando numéricos por rangos)
        # ──────────────────────────────────────────────
        stage4 = grouped.filter(pl.col("StageSequence") == 4)
        stage8 = grouped.filter(pl.col("StageSequence") == 8)

        # Definir rangos para Distance, MeasuredTonnage y ReportedTonnage
        distance_bins = [500, 1000, 1500, 2000, 2500, 3000]
        distance_labels = [
            "0-500",
            "500-1000",
            "1000-1500",
            "1500-2000",
            "2000-2500",
            "2500-3000",
            "3000+",
        ]

        tonnage_bins = [50, 100, 150, 200, 250, 300]
        tonnage_labels = [
            "0-50",
            "50-100",
            "100-150",
            "150-200",
            "200-250",
            "250-300",
            "300+",
        ]

        # Crear columnas de rango
        stage4 = stage4.with_columns(
            pl.col("Distance")
            .cut(
                breaks=distance_bins,
                labels=distance_labels,
                include_breaks=False,
            )
            .alias("DistanceRange")
        )

        stage8 = stage8.with_columns(
            [
                pl.col("Distance")
                .cut(
                    breaks=distance_bins,
                    labels=distance_labels,
                    include_breaks=False,
                )
                .alias("DistanceRange"),
                pl.col("MeasuredTonnage")
                .cut(
                    breaks=tonnage_bins,
                    labels=tonnage_labels,
                    include_breaks=False,
                )
                .alias("MeasuredTonnageRange"),
                pl.col("ReportedTonnage")
                .cut(
                    breaks=tonnage_bins,
                    labels=tonnage_labels,
                    include_breaks=False,
                )
                .alias("ReportedTonnageRange"),
            ]
        )

        # Eficiencia Stage 4
        stats["efficiency_stage4_factors"] = {
            "Shovel": stage4.group_by("Shovel")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "ShovelModel": stage4.group_by("ShovelModel")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "LoadingZone": stage4.group_by("LoadingZone")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "DistanceRange": stage4.group_by("DistanceRange")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
        }

        # Eficiencia Stage 8
        stats["efficiency_stage8_factors"] = {
            "Material": stage8.group_by("Material")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "MeasuredTonnageRange": stage8.group_by("MeasuredTonnageRange")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "ReportedTonnageRange": stage8.group_by("ReportedTonnageRange")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "DestinationType": stage8.group_by("DestinationType")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "Destination": stage8.group_by("Destination")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
            "DistanceRange": stage8.group_by("DistanceRange")
            .agg(pl.col("TotalEfficiency").mean().alias("AvgEfficiency"))
            .to_dicts(),
        }

        # ──────────────────────────────────────────────
        # 5️⃣ Eficiencia global en final vacío (4) y final lleno (8)
        # ──────────────────────────────────────────────
        stats["overall_efficiency"] = {
            "StageSequence_4": stage4["TotalEfficiency"].mean(),
            "StageSequence_8": stage8["TotalEfficiency"].mean(),
        }

        # ──────────────────────────────────────────────
        # 6️⃣ Eficiencia media por tiempo (día, hora, mes) y estado
        # ──────────────────────────────────────────────
        df = df.with_columns(
            [
                pl.col("TimeStampIni").dt.hour().alias("Hour"),
                pl.col("TimeStampIni").dt.weekday().alias("Weekday"),
                pl.col("TimeStampIni").dt.month().alias("Month"),
            ]
        )

        # Agrupamientos
        weekday_eff = (
            df.group_by(["Weekday", "OperationGroup"])
            .agg(pl.col("TimeEfficiencyPercentage").mean().alias("AvgEfficiency"))
            .sort(["Weekday", "OperationGroup"])
            .to_dicts()
        )

        hour_eff = (
            df.group_by(["Hour", "OperationGroup"])
            .agg(pl.col("TimeEfficiencyPercentage").mean().alias("AvgEfficiency"))
            .sort(["Hour", "OperationGroup"])
            .to_dicts()
        )

        month_eff = (
            df.group_by(["Month", "OperationGroup"])
            .agg(pl.col("TimeEfficiencyPercentage").mean().alias("AvgEfficiency"))
            .sort(["Month", "OperationGroup"])
            .to_dicts()
        )

        stats["efficiency_by_time"] = {
            "weekday_efficiency": weekday_eff,
            "hour_efficiency": hour_eff,
            "month_efficiency": month_eff,
        }

        return stats

    def run(self):
        """Ejecutar análisis completo"""
        self._load_cycle_data()
        self._generate_statistics()

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estadísticas generadas"""
        if not self._stats_generated:
            raise RuntimeError("Primero ejecute run()")
        return self._stats_cache

    def close(self):
        """Cerrar conexión a ClickHouse"""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception as e:
                logging.warning(f"Error cerrando conexión: {e}")


if __name__ == "__main__":
    eda = CycleDataEDA(truck_id="T-210")
    eda.run()
    stats = eda.analyze_time_efficiency()
    eda.close()
