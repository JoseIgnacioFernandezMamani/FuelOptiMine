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
            FROM timemodel_table
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
                    pl.col("TimeStamp_tm")
                    .diff()
                    .dt.total_seconds()
                    .alias("StateDurationSeconds"),
                    # Componentes temporales
                    pl.col("TimeStamp_tm").dt.hour().alias("Hour"),
                    pl.col("TimeStamp_tm").dt.date().alias("Date"),
                    pl.col("TimeStamp_tm").dt.month().alias("Month"),
                    pl.col("TimeStamp_tm").dt.weekday().alias("Weekday"),
                    pl.col("TimeStamp_tm").dt.year().alias("Year"),
                    # Shift aproximado (simplificado)
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

        for col in df.columns:
            col_type = df.schema[col]
            col_stats = {}

            # Variables categóricas
            if col in self.categorical_cols or (
                col_type == pl.Utf8 and col not in self.id_cols
            ):
                value_counts = df[col].value_counts().sort("count", descending=True)
                unique_count = df[col].n_unique()
                total = len(df)

                col_stats = {
                    "type": "categorical",
                    "unique_values": unique_count,
                    "total_records": total,
                    "most_common": value_counts.head(10).to_dicts(),
                    "null_count": df[col].null_count(),
                    "null_percentage": (
                        (df[col].null_count() / total) * 100 if total > 0 else 0
                    ),
                }

            # Variables numéricas (duración)
            elif col == "StateDurationSeconds":
                # Filtrar valores válidos para estadísticas
                valid_durations = df[col].drop_nulls()

                if len(valid_durations) > 0:
                    col_stats = {
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
                        "null_count": df[col].null_count(),
                    }

            # Variables datetime
            elif col_type in [pl.Datetime]:
                min_date = df[col].min()
                max_date = df[col].max()

                col_stats = {
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

            stats[col] = col_stats

        self._stats_generated = True
        self._stats_cache = stats
        return stats

    def get_state_transitions(self) -> pl.DataFrame:
        """
        Analizar transiciones entre estados.
        Muestra qué estado sigue a qué estado.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        transitions = self.timemodel_df.with_columns(
            [
                pl.col("Status").shift(-1).alias("NextStatus"),
                pl.col("Category").shift(-1).alias("NextCategory"),
            ]
        ).filter(pl.col("NextStatus").is_not_null())

        transition_counts = (
            transitions.group_by(["Status", "NextStatus"])
            .agg([pl.count().alias("count")])
            .sort("count", descending=True)
        )

        # Calcular porcentajes
        total = transition_counts["count"].sum()
        transition_counts = transition_counts.with_columns(
            [(pl.col("count") / total * 100).alias("percentage")]
        )

        return transition_counts

    def get_state_duration_analysis(self, state_col: str = "Status") -> pl.DataFrame:
        """
        Análisis de duración de estados/categorías/eventos.

        Args:
            state_col: 'Status', 'Category', o 'Event'
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        if state_col not in self.categorical_cols:
            raise ValueError(f"'{state_col}' debe ser una de: {self.categorical_cols}")

        duration_analysis = (
            self.timemodel_df.filter(pl.col("StateDurationSeconds").is_not_null())
            .group_by(state_col)
            .agg(
                [
                    pl.count().alias("occurrences"),
                    pl.col("StateDurationSeconds").sum().alias("total_seconds"),
                    pl.col("StateDurationSeconds").mean().alias("avg_seconds"),
                    pl.col("StateDurationSeconds").median().alias("median_seconds"),
                    pl.col("StateDurationSeconds").min().alias("min_seconds"),
                    pl.col("StateDurationSeconds").max().alias("max_seconds"),
                    pl.col("StateDurationSeconds").std().alias("std_seconds"),
                ]
            )
            .with_columns(
                [
                    (pl.col("total_seconds") / 3600).alias("total_hours"),
                    (pl.col("avg_seconds") / 60).alias("avg_minutes"),
                    (pl.col("median_seconds") / 60).alias("median_minutes"),
                ]
            )
            .sort("total_hours", descending=True)
        )

        return duration_analysis

    def get_temporal_distribution(
        self, state_col: str = "Status", time_group: str = "hour"
    ) -> pl.DataFrame:
        """
        Distribución temporal de estados.

        Args:
            state_col: Columna a analizar ('Status', 'Category', 'Event')
            time_group: 'hour', 'weekday', 'month', 'shift'
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        time_mapping = {
            "hour": "Hour",
            "weekday": "Weekday",
            "month": "Month",
            "shift": "Shift",
        }

        group_col = time_mapping.get(time_group, "Hour")

        distribution = (
            self.timemodel_df.group_by([group_col, state_col])
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("StateDurationSeconds")
                    .sum()
                    .alias("total_duration_seconds"),
                ]
            )
            .sort([group_col, "count"], descending=[False, True])
        )

        # Agregar porcentaje por grupo temporal
        total_by_time = distribution.group_by(group_col).agg(
            [pl.col("count").sum().alias("time_total")]
        )

        distribution = (
            distribution.join(total_by_time, on=group_col)
            .with_columns(
                [(pl.col("count") / pl.col("time_total") * 100).alias("percentage")]
            )
            .drop("time_total")
        )

        return distribution

    def get_uptime_downtime_analysis(self) -> Dict[str, Any]:
        """
        Análisis de tiempo operativo vs no operativo.
        Requiere definir qué estados son "operativos" según tu negocio.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        df = self.timemodel_df

        # Estados operativos comunes (ajustar según tu caso)
        operational_statuses = [
            "Operating",
            "Running",
            "Active",
            "Working",
            "Loaded",
            "Empty",
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
        }

    def get_event_frequency_analysis(self) -> pl.DataFrame:
        """
        Análisis de frecuencia de eventos.
        Útil para identificar eventos recurrentes o problemáticos.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        event_freq = (
            self.timemodel_df.group_by(["Category", "Event"])
            .agg(
                [
                    pl.count().alias("occurrences"),
                    pl.col("StateDurationSeconds").mean().alias("avg_duration_seconds"),
                ]
            )
            .with_columns(
                [(pl.col("avg_duration_seconds") / 60).alias("avg_duration_minutes")]
            )
            .sort("occurrences", descending=True)
        )

        return event_freq

    def get_longest_states(self, top_n: int = 20) -> pl.DataFrame:
        """
        Identificar los estados más largos registrados.
        Útil para detectar anomalías o periodos críticos.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        longest = (
            self.timemodel_df.filter(pl.col("StateDurationSeconds").is_not_null())
            .select(
                [
                    "TimeStamp_tm",
                    "Status",
                    "Category",
                    "Event",
                    "StateDurationSeconds",
                    "Shift",
                    "Date",
                ]
            )
            .with_columns(
                [(pl.col("StateDurationSeconds") / 3600).alias("DurationHours")]
            )
            .sort("StateDurationSeconds", descending=True)
            .head(top_n)
        )

        return longest

    def get_daily_summary(self) -> pl.DataFrame:
        """
        Resumen diario de estados y tiempo.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        daily = (
            self.timemodel_df.filter(pl.col("StateDurationSeconds").is_not_null())
            .group_by("Date")
            .agg(
                [
                    pl.count().alias("total_events"),
                    pl.col("Status").n_unique().alias("unique_statuses"),
                    pl.col("StateDurationSeconds").sum().alias("total_seconds"),
                    pl.col("StateDurationSeconds").mean().alias("avg_event_duration"),
                ]
            )
            .with_columns([(pl.col("total_seconds") / 3600).alias("total_hours")])
            .sort("Date")
        )

        return daily

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


if __name__ == "__main__":
    # Ejemplo de uso
    eda = TimeModelDataEDA(truck_id="T-243")
    eda.run()

    df = eda.get_dataframe()
    stats = eda.get_statistics()

    print("=" * 60)
    print("ESTADÍSTICAS GENERALES")
    print("=" * 60)
    print(f"Total de registros: {len(df)}")
    print(f"\nEstados únicos: {df['Status'].n_unique()}")
    print(f"Categorías únicas: {df['Category'].n_unique()}")
    print(f"Eventos únicos: {df['Event'].n_unique()}")

    print("\n" + "=" * 60)
    print("ANÁLISIS DE DURACIÓN POR ESTADO")
    print("=" * 60)
    duration_by_status = eda.get_state_duration_analysis("Status")
    print(duration_by_status)

    print("\n" + "=" * 60)
    print("TRANSICIONES DE ESTADO MÁS COMUNES")
    print("=" * 60)
    transitions = eda.get_state_transitions()
    print(transitions.head(10))

    print("\n" + "=" * 60)
    print("ANÁLISIS UPTIME/DOWNTIME")
    print("=" * 60)
    uptime_analysis = eda.get_uptime_downtime_analysis()
    print(f"Total horas: {uptime_analysis['total_hours']:.2f}")
    print(f"Uptime: {uptime_analysis['uptime_percentage']:.2f}%")
    print(f"Downtime: {uptime_analysis['downtime_percentage']:.2f}%")

    print("\n" + "=" * 60)
    print("EVENTOS MÁS FRECUENTES")
    print("=" * 60)
    event_freq = eda.get_event_frequency_analysis()
    print(event_freq.head(10))

    print("\n" + "=" * 60)
    print("ESTADOS MÁS LARGOS")
    print("=" * 60)
    longest = eda.get_longest_states(top_n=10)
    print(longest)

    eda.close()
