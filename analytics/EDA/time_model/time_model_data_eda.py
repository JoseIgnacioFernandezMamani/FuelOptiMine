import polars as pl
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging
from etl_core.load.utils import create_client, CH_CONFIG


class TimeModelDataEDA:
    """
    Análisis exploratorio de datos de Time Model.
    Enfocado en estados, categorías y eventos de equipos mineros.
    """

    def __init__(self, truck_id: str) -> None:
        self.timemodel_df: pl.DataFrame = pl.DataFrame()
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._data_loaded: bool = False
        self._stats_generated: bool = False
        self.truck_id = truck_id
        self.client = create_client(CH_CONFIG)

        # Definición de columnas
        self.categorical_cols = ["Status", "Category", "Event"]
        self.datetime_cols = ["TimeStamp_tm"]
        self.id_cols = ["TimeModelId"]

    def _load_timemodel_data(self):
        """Cargar datos de Time Model desde ClickHouse"""
        try:
            query = f"""
            SELECT 
                TimeModelId,
                TimeStamp_tm,
                Status,
                Category,
                Event
            FROM xgboost_fuel
            WHERE Equipment = '{self.truck_id}'
            AND TimeModelId IS NOT NULL
            ORDER BY TimeStamp_tm
            """

            pandas_df = self.client.query_df(query)

            if pandas_df.empty:
                raise RuntimeError(
                    f"No se encontraron datos de Time Model para el equipo {self.truck_id}"
                )

            self.timemodel_df = pl.from_pandas(pandas_df)

            # Calcular columnas derivadas
            self.timemodel_df = self.timemodel_df.with_columns(
                [
                    # Duración de cada estado (diferencia con el siguiente timestamp)
                    (pl.col("TimeStamp_tm").shift(-1) - pl.col("TimeStamp_tm"))
                    .dt.total_seconds()
                    .alias("StateDurationSeconds"),
                    # Componentes temporales
                    pl.col("TimeStamp_tm").dt.hour().alias("Hour"),
                    pl.col("TimeStamp_tm").dt.date().alias("Date"),
                    pl.col("TimeStamp_tm").dt.month().alias("Month"),
                    pl.col("TimeStamp_tm").dt.weekday().alias("Weekday"),
                    pl.col("TimeStamp_tm").dt.year().alias("Year"),
                    # Shift aproximado
                    pl.when(pl.col("TimeStamp_tm").dt.hour().is_between(7, 19))
                    .then(pl.lit("D"))
                    .otherwise(pl.lit("N"))
                    .alias("Shift"),
                ]
            )

            # Eliminar valores negativos o extremos en duración (posibles errores)
            self.timemodel_df = self.timemodel_df.with_columns(
                [
                    pl.when(
                        (pl.col("StateDurationSeconds") > 0)
                        & (pl.col("StateDurationSeconds") < 86400)  # Menos de 24 horas
                    )
                    .then(pl.col("StateDurationSeconds"))
                    .otherwise(None)
                    .alias("StateDurationSeconds")
                ]
            )

            self._data_loaded = True
            self._stats_cache = {}
            self._stats_generated = False

        except Exception as e:
            raise RuntimeError(f"Error al cargar datos de Time Model: {str(e)}")

    def _generate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Generar estadísticas para variables categóricas y temporal"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        stats = {}
        df = self.timemodel_df

        # Estadísticas de TimeModelId (solo conteo)
        if "TimeModelId" in df.columns:
            stats["TimeModelId"] = {
                "type": "identifier",
                "total_events": df["TimeModelId"].n_unique(),
                "total_records": len(df),
            }

        # Variables categóricas
        for col in self.categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts().sort("count", descending=True)
                unique_count = df[col].n_unique()
                total = len(df)

                stats[col] = {
                    "type": "categorical",
                    "unique_values": unique_count,
                    "total_records": total,
                    "top_10_values": value_counts.head(10).to_dicts(),
                    "all_values": value_counts.to_dicts(),
                    "null_count": df[col].null_count(),
                    "null_percentage": (
                        (df[col].null_count() / total) * 100 if total > 0 else 0
                    ),
                }

        # StateDurationSeconds (numérica derivada)
        if "StateDurationSeconds" in df.columns:
            valid_durations = df["StateDurationSeconds"].drop_nulls()

            if len(valid_durations) > 0:
                stats["StateDurationSeconds"] = {
                    "type": "numeric",
                    "mean": valid_durations.mean(),
                    "median": valid_durations.median(),
                    "min": valid_durations.min(),
                    "max": valid_durations.max(),
                    "std_dev": valid_durations.std(),
                    "q1": valid_durations.quantile(0.25),
                    "q3": valid_durations.quantile(0.75),
                    "p5": valid_durations.quantile(0.05),
                    "p95": valid_durations.quantile(0.95),
                    "valid_count": len(valid_durations),
                    "null_count": df["StateDurationSeconds"].null_count(),
                    "mean_minutes": valid_durations.mean() / 60,
                    "mean_hours": valid_durations.mean() / 3600,
                }

        # TimeStamp_tm
        if "TimeStamp_tm" in df.columns:
            min_date = df["TimeStamp_tm"].min()
            max_date = df["TimeStamp_tm"].max()

            stats["TimeStamp_tm"] = {
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
                "total_records": len(df),
            }

        # Distribuciones temporales
        if "Hour" in df.columns:
            stats["Hour_distribution"] = sorted(
                df["Hour"].value_counts().to_dicts(), key=lambda x: x["Hour"]
            )

        if "Weekday" in df.columns:
            stats["Weekday_distribution"] = sorted(
                df["Weekday"].value_counts().to_dicts(), key=lambda x: x["Weekday"]
            )

        if "Month" in df.columns:
            stats["Month_distribution"] = sorted(
                df["Month"].value_counts().to_dicts(), key=lambda x: x["Month"]
            )

        if "Shift" in df.columns:
            stats["Shift_distribution"] = df["Shift"].value_counts().to_dicts()

        self._stats_generated = True
        self._stats_cache = stats
        return stats

    def analyze_state_patterns(self) -> Dict[str, Any]:
        """
        Análisis completo de patrones de estados.
        """
        df = self.timemodel_df
        analysis = {}

        # 1. Transiciones de estados
        transitions = df.with_columns(
            [
                pl.col("Status").shift(-1).alias("NextStatus"),
                pl.col("StateDurationSeconds").alias("CurrentDuration"),
            ]
        ).filter(pl.col("NextStatus").is_not_null())

        transition_counts = (
            transitions.group_by(["Status", "NextStatus"])
            .agg([pl.count().alias("count")])
            .sort("count", descending=True)
        )

        total_transitions = transition_counts["count"].sum()
        transition_counts = transition_counts.with_columns(
            [(pl.col("count") / total_transitions * 100).alias("percentage")]
        )

        analysis["state_transitions"] = {
            "top_20_transitions": transition_counts.head(20).to_dicts(),
            "total_unique_transitions": len(transition_counts),
        }

        # 2. Duración por estado
        duration_by_status = (
            df.filter(pl.col("StateDurationSeconds").is_not_null())
            .group_by("Status")
            .agg(
                [
                    pl.count().alias("occurrences"),
                    pl.col("StateDurationSeconds").sum().alias("total_seconds"),
                    pl.col("StateDurationSeconds").mean().alias("avg_seconds"),
                    pl.col("StateDurationSeconds").median().alias("median_seconds"),
                    pl.col("StateDurationSeconds").std().alias("std_seconds"),
                ]
            )
            .with_columns(
                [
                    (pl.col("total_seconds") / 3600).alias("total_hours"),
                    (pl.col("avg_seconds") / 60).alias("avg_minutes"),
                ]
            )
            .sort("total_hours", descending=True)
        )

        analysis["duration_by_status"] = duration_by_status.to_dicts()

        # 3. Duración por categoría
        duration_by_category = (
            df.filter(pl.col("StateDurationSeconds").is_not_null())
            .group_by("Category")
            .agg(
                [
                    pl.count().alias("occurrences"),
                    pl.col("StateDurationSeconds").sum().alias("total_seconds"),
                    pl.col("StateDurationSeconds").mean().alias("avg_seconds"),
                ]
            )
            .with_columns([(pl.col("total_seconds") / 3600).alias("total_hours")])
            .sort("total_hours", descending=True)
        )

        analysis["duration_by_category"] = duration_by_category.to_dicts()

        # 4. Eventos más frecuentes
        event_frequency = (
            df.group_by(["Category", "Event"])
            .agg([pl.count().alias("occurrences")])
            .sort("occurrences", descending=True)
        )

        analysis["event_frequency"] = {
            "top_20_events": event_frequency.head(20).to_dicts(),
            "total_unique_events": len(event_frequency),
        }

        # 5. Estados más largos registrados
        longest_states = (
            df.filter(pl.col("StateDurationSeconds").is_not_null())
            .select(
                [
                    "TimeStamp_tm",
                    "Status",
                    "Category",
                    "Event",
                    "StateDurationSeconds",
                ]
            )
            .with_columns(
                [(pl.col("StateDurationSeconds") / 3600).alias("DurationHours")]
            )
            .sort("StateDurationSeconds", descending=True)
            .head(20)
        )

        analysis["longest_states"] = longest_states.to_dicts()

        return analysis

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """
        Análisis de patrones temporales.
        """
        df = self.timemodel_df
        analysis = {}

        # 1. Distribución por hora del día
        hourly_status = (
            df.group_by(["Hour", "Status"])
            .agg([pl.count().alias("count")])
            .sort(["Hour", "count"], descending=[False, True])
        )

        analysis["hourly_status_distribution"] = hourly_status.to_dicts()

        # 2. Distribución por día de la semana
        weekday_status = (
            df.group_by(["Weekday", "Status"])
            .agg([pl.count().alias("count")])
            .sort(["Weekday", "count"], descending=[False, True])
        )

        analysis["weekday_status_distribution"] = weekday_status.to_dicts()

        # 3. Distribución por turno
        shift_status = (
            df.group_by(["Shift", "Status"])
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("StateDurationSeconds").sum().alias("total_duration"),
                ]
            )
            .sort(["Shift", "count"], descending=[False, True])
        )

        analysis["shift_status_distribution"] = shift_status.to_dicts()

        # 4. Resumen diario
        daily_summary = (
            df.group_by("Date")
            .agg(
                [
                    pl.count().alias("total_events"),
                    pl.col("Status").n_unique().alias("unique_statuses"),
                    pl.col("StateDurationSeconds")
                    .sum()
                    .alias("total_duration_seconds"),
                ]
            )
            .with_columns(
                [(pl.col("total_duration_seconds") / 3600).alias("total_hours")]
            )
            .sort("Date")
        )

        analysis["daily_summary"] = daily_summary.to_dicts()

        # 5. Eventos por mes
        monthly_events = (
            df.group_by("Month").agg([pl.count().alias("total_events")]).sort("Month")
        )

        analysis["monthly_events"] = monthly_events.to_dicts()

        return analysis

    def analyze_uptime_downtime(
        self, operational_statuses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Análisis de tiempo operativo vs no operativo.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        df = self.timemodel_df

        # Estados operativos por defecto (personalizar según necesidad)
        if operational_statuses is None:
            operational_statuses = [
                "Operating",
                "Running",
                "Active",
                "Working",
                "Loaded",
                "Empty",
                "Hauling",
                "Loading",
            ]

        # Clasificar estados
        classified = df.with_columns(
            [
                pl.when(pl.col("Status").is_in(operational_statuses))
                .then(pl.lit("Uptime"))
                .otherwise(pl.lit("Downtime"))
                .alias("TimeType")
            ]
        )

        # Calcular totales
        time_summary = (
            classified.filter(pl.col("StateDurationSeconds").is_not_null())
            .group_by("TimeType")
            .agg(
                [
                    pl.count().alias("events"),
                    pl.col("StateDurationSeconds").sum().alias("total_seconds"),
                ]
            )
            .with_columns([(pl.col("total_seconds") / 3600).alias("total_hours")])
        )

        total_hours = time_summary["total_hours"].sum()

        uptime_row = time_summary.filter(pl.col("TimeType") == "Uptime")
        downtime_row = time_summary.filter(pl.col("TimeType") == "Downtime")

        uptime_hours = uptime_row["total_hours"][0] if len(uptime_row) > 0 else 0
        downtime_hours = downtime_row["total_hours"][0] if len(downtime_row) > 0 else 0

        # Distribución de downtime por status
        downtime_by_status = (
            classified.filter(
                (pl.col("TimeType") == "Downtime")
                & (pl.col("StateDurationSeconds").is_not_null())
            )
            .group_by("Status")
            .agg(
                [
                    pl.count().alias("occurrences"),
                    pl.col("StateDurationSeconds").sum().alias("total_seconds"),
                ]
            )
            .with_columns([(pl.col("total_seconds") / 3600).alias("total_hours")])
            .sort("total_hours", descending=True)
        )

        return {
            "total_hours": float(total_hours),
            "uptime_hours": float(uptime_hours),
            "downtime_hours": float(downtime_hours),
            "uptime_percentage": (
                (uptime_hours / total_hours * 100) if total_hours > 0 else 0
            ),
            "downtime_percentage": (
                (downtime_hours / total_hours * 100) if total_hours > 0 else 0
            ),
            "summary": time_summary.to_dicts(),
            "downtime_by_status": downtime_by_status.to_dicts(),
        }

    def analyze_category_event_relationship(self) -> Dict[str, Any]:
        """
        Análisis de relación entre categorías y eventos.
        """
        df = self.timemodel_df
        analysis = {}

        # 1. Eventos por categoría
        events_per_category = (
            df.group_by(["Category", "Event"])
            .agg(
                [
                    pl.count().alias("occurrences"),
                    pl.col("StateDurationSeconds").mean().alias("avg_duration_seconds"),
                ]
            )
            .sort(["Category", "occurrences"], descending=[False, True])
        )

        analysis["events_per_category"] = events_per_category.to_dicts()

        # 2. Categorías más frecuentes por Status
        status_category = (
            df.group_by(["Status", "Category"])
            .agg([pl.count().alias("occurrences")])
            .sort(["Status", "occurrences"], descending=[False, True])
        )

        analysis["status_category_relationship"] = status_category.to_dicts()

        # 3. Matriz de frecuencia Status-Category
        status_list = df["Status"].unique().to_list()
        category_list = df["Category"].unique().to_list()

        analysis["unique_statuses"] = status_list
        analysis["unique_categories"] = category_list

        return analysis

    def run(self):
        """Ejecutar análisis completo"""
        self._load_timemodel_data()
        self._generate_statistics()

    def get_dataframe(self) -> pl.DataFrame:
        """Obtener DataFrame cargado"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")
        return self.timemodel_df

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
