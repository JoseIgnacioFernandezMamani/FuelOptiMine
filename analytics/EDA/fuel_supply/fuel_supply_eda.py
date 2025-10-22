from typing import Any, Dict, Optional
import polars as pl
import numpy as np
from datetime import datetime, date
import logging
from etl_core.load.utils import create_client, CH_CONFIG
from .sensor_supply_event_correlator import SensorSupplyEventCorrelator


class FuelSupplyEDA:
    """
    Análisis exploratorio de datos para la tabla fuel_supply.
    Enfocado en eventos de recarga de combustible.
    """

    def __init__(self, truck_id: Optional[str] = "T-210") -> None:
        """
        Args:
            equipment: Filtrar por equipo específico (ej: 'T-210')
        """
        self.supply_df: pl.DataFrame = pl.DataFrame()
        self.sensor_df: pl.DataFrame = pl.DataFrame()
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._data_loaded: bool = False
        self._stats_generated: bool = False
        self.truck_id = truck_id
        self.client = create_client(CH_CONFIG)

    def get_dataframe(self) -> pl.DataFrame:
        """Obtener el DataFrame cargado"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")
        return self.supply_df

    def _load_fuel_supply_data(self):
        """Cargar datos de recarga desde ClickHouse"""
        try:
            # Construir filtros dinámicos

            query = f"""
            SELECT 
                FuelSupplyId,
                TimeStamp,
                ShiftDate,
                Shift,
                Origin,
                Equipment,
                TruckFleet,
                FuelLevelLiters,
                FuelLevel
            FROM fuel_optimine.fuel_supply
            WHERE Equipment = '{self.truck_id}'
            ORDER BY TimeStamp
            """

            pandas_df = self.client.query_df(query)

            # Convertir a Polars
            self.supply_df = pl.from_pandas(pandas_df)

            self.supply_df = self.supply_df.with_columns(
                pl.col("TimeStamp")
                .dt.truncate("1ms")
                .cast(pl.Datetime("ms"))
                .alias("TimeStamp")
            )

            # Calcular métricas derivadas
            self.supply_df = self.supply_df.with_columns(
                [
                    # Diferencia de tiempo entre registros consecutivos
                    pl.col("TimeStamp")
                    .diff()
                    .dt.total_seconds()
                    .alias("TimeDiffSeconds"),
                    # Diferencia de nivel de combustible (detecta recargas)
                    pl.col("FuelLevelLiters").diff().alias("DeltaFuelLiters"),
                    # Extrae hora del día para análisis temporal
                    pl.col("TimeStamp").dt.hour().alias("Hour"),
                    # Día de la semana (0=Lunes, 6=Domingo)
                    pl.col("TimeStamp").dt.weekday().alias("Weekday"),
                ]
            )

            # Filtrar valores de combustible válidos
            self.supply_df = self.supply_df.filter(pl.col("FuelLevelLiters") >= 0)

            self._data_loaded = True
            self._stats_cache = {}
            self._stats_generated = False

            logging.info(f"✅ Cargados {len(self.supply_df)} registros de fuel_supply")

        except Exception as e:
            logging.error(f"❌ Error al cargar datos: {e}")
            raise RuntimeError(f"Error al cargar datos de fuel_supply: {e}")

    def _load_refuel_events(self):
        try:

            query = f"""
            SELECT 
                TimeStamp,
                TruckFleet,
                ValidFuel,
                DeltaFuel,
                BeforeAvg,
                AfterAvg
            FROM xgboost_fuel
            WHERE Equipment = '{self.truck_id}' AND ValidFuel > 0
            ORDER BY TimeStamp
            """
            # obtener los datos de clickhouse
            pandas_df = self.client.query_df(query)

            if pandas_df.empty:
                raise RuntimeError(
                    f"No se encontraron datos para el equipo {self.truck_id}"
                )

            # convertir a polars
            self.sensor_df = pl.from_pandas(pandas_df)
            self.sensor_df = self.sensor_df.with_columns(
                pl.col("TimeStamp")
                .dt.truncate("1ms")
                .cast(pl.Datetime("ms"))
                .alias("TimeStamp")
            )

            logging.info(
                f"✅ Cargados {len(self.sensor_df)} eventos de recarga desde xgboost_fuel"
            )

        except FileNotFoundError:
            raise RuntimeError(
                f"Error para cargar los datos del equipo {self.truck_id}"
            )

    def correlate_supply_events(
        self, start_date: date = date(2024, 2, 1), end_date: date = date(2025, 2, 28)
    ) -> pl.DataFrame:
        """
        Correlaciona eventos de sensores con eventos de suministro de combustible.

        Args:
            start_date: Fecha de inicio del análisis (opcional)
            end_date: Fecha de fin del análisis (opcional)

        Returns:
            pl.DataFrame: DataFrame con eventos correlacionados

        Raises:
            RuntimeError: Si no se han cargado los datos o si falla la correlación
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        try:

            # Inicializar correlator
            correlator = SensorSupplyEventCorrelator(truck_id=self.truck_id)

            # Asignar dataframes al correlator
            correlator.load_datasets(
                refill_df=self.sensor_df, fuel_supply_df=self.supply_df
            )

            # obtener las fechas dinamicamente

            correlator.filter_by_date_range(start_date, end_date)

            # Preparar datos
            correlator._prepare()

            # Correlacionar eventos
            correlated_df = correlator.correlate_events()

            logging.info(f"✅ Correlación completada: {len(correlated_df)} eventos")

            return correlated_df

        except Exception as e:
            logging.error(f"❌ Error en correlación: {e}")
            raise RuntimeError(f"Error al correlacionar eventos: {str(e)}")

    def _generate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Generar estadísticas descriptivas básicas"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute _load_fuel_supply_data()")

        stats = {}
        df = self.supply_df

        # Columnas numéricas para análisis
        numeric_cols = [
            "FuelLevelLiters",
            "FuelLevel",
            "TimeDiffSeconds",
            "DeltaFuelLiters",
        ]

        for col in numeric_cols:
            if col not in df.columns:
                continue

            col_data = df[col].drop_nulls()

            if len(col_data) == 0:
                stats[col] = {"error": "No hay datos válidos"}
                continue

            stats[col] = {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "std": float(col_data.std()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75)),
                "null_count": df[col].null_count(),
            }

        # Estadísticas temporales
        stats["temporal"] = {
            "first_record": df["TimeStamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "last_record": df["TimeStamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
            "total_days": (df["ShiftDate"].max() - df["ShiftDate"].min()).days,
            "unique_dates": df["ShiftDate"].n_unique(),
        }

        self._stats_generated = True
        self._stats_cache = stats
        return stats

    def run(self):
        """Ejecutar el análisis completo"""
        self._load_fuel_supply_data()
        self._load_refuel_events()
        self._generate_statistics()

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estadísticas generadas"""
        if not self._stats_generated:
            raise RuntimeError("Primero ejecute run()")
        return self._stats_cache

    def analyze_refuel_events(self) -> Dict[str, Any]:
        """
        Detectar y analizar eventos de recarga de combustible.
        Un evento de recarga se identifica cuando DeltaFuelLiters > umbral.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        # Umbral mínimo para considerar una recarga (litros)
        REFUEL_THRESHOLD = 100  # Ajustar según características de la operación

        refuels = self.supply_df.filter(
            (pl.col("DeltaFuelLiters").is_not_null())
            & (pl.col("DeltaFuelLiters") > REFUEL_THRESHOLD)
        )

        if refuels.is_empty():
            return {
                "total_refuels": 0,
                "message": "No se detectaron eventos de recarga",
            }

        # Estadísticas generales
        total_refuels = len(refuels)
        total_volume = float(refuels["DeltaFuelLiters"].sum())
        avg_volume = float(refuels["DeltaFuelLiters"].mean())

        # Distribución por turno
        by_shift = (
            refuels.group_by("Shift")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("DeltaFuelLiters").sum().alias("total_volume"),
                    pl.col("DeltaFuelLiters").mean().alias("avg_volume"),
                ]
            )
            .sort("Shift")
        )

        # Distribución por origen de recarga
        by_origin = (
            refuels.group_by("Origin")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("DeltaFuelLiters").sum().alias("total_volume"),
                    pl.col("DeltaFuelLiters").mean().alias("avg_volume"),
                ]
            )
            .sort("count", descending=True)
        )

        # Distribución por equipo (top 10)
        by_equipment = (
            refuels.group_by("Equipment")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("DeltaFuelLiters").sum().alias("total_volume"),
                ]
            )
            .sort("total_volume", descending=True)
            .head(10)
        )

        # Distribución por flota
        by_fleet = (
            refuels.group_by("TruckFleet")
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("DeltaFuelLiters").sum().alias("total_volume"),
                    pl.col("DeltaFuelLiters").mean().alias("avg_volume"),
                ]
            )
            .sort("total_volume", descending=True)
        )

        # Distribución temporal (por mes)
        by_month = (
            refuels.with_columns(
                [
                    pl.col("ShiftDate").dt.month().alias("Month"),
                    pl.col("ShiftDate").dt.year().alias("Year"),
                ]
            )
            .group_by(["Year", "Month"])
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col("DeltaFuelLiters").sum().alias("total_volume"),
                ]
            )
            .sort(["Year", "Month"])
        )

        # Análisis por hora del día
        by_hour = refuels.group_by("Hour").agg([pl.count().alias("count")]).sort("Hour")

        return {
            "total_refuels": total_refuels,
            "total_volume_liters": total_volume,
            "avg_volume_liters": avg_volume,
            "refuels_by_shift": by_shift.to_dicts(),
            "refuels_by_origin": by_origin.to_dicts(),
            "refuels_by_equipment": by_equipment.to_dicts(),
            "refuels_by_fleet": by_fleet.to_dicts(),
            "refuels_by_month": by_month.to_dicts(),
            "refuels_by_hour": by_hour.to_dicts(),
            "refuel_events_df": refuels,  # DataFrame para análisis detallado
        }

    def get_fleet_summary(self) -> Dict[str, Any]:
        """
        Resumen por flota de camiones: conteo de equipos,
        total de registros y estadísticas de combustible.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        summary = (
            self.supply_df.group_by("TruckFleet")
            .agg(
                [
                    pl.col("Equipment").n_unique().alias("unique_equipment"),
                    pl.count().alias("total_records"),
                    pl.col("FuelLevelLiters").mean().alias("avg_fuel_level"),
                    pl.col("FuelLevelLiters").max().alias("max_fuel_level"),
                    pl.col("FuelLevelLiters").min().alias("min_fuel_level"),
                ]
            )
            .sort("total_records", descending=True)
        )

        return {"fleet_summary": summary.to_dicts(), "total_fleets": len(summary)}

    def get_origin_summary(self) -> Dict[str, Any]:
        """
        Resumen por origen de suministro: frecuencia de uso
        y volumen total suministrado.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        summary = (
            self.supply_df.group_by("Origin")
            .agg(
                [
                    pl.count().alias("total_records"),
                    pl.col("Equipment").n_unique().alias("unique_equipment"),
                    pl.col("FuelLevelLiters").mean().alias("avg_fuel_level"),
                ]
            )
            .sort("total_records", descending=True)
        )

        return {"origin_summary": summary.to_dicts(), "total_origins": len(summary)}

    def get_shift_analysis(self) -> Dict[str, Any]:
        """
        Análisis comparativo entre turnos día/noche.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        shift_stats = (
            self.supply_df.group_by("Shift")
            .agg(
                [
                    pl.count().alias("total_records"),
                    pl.col("Equipment").n_unique().alias("unique_equipment"),
                    pl.col("FuelLevelLiters").mean().alias("avg_fuel_level"),
                    pl.col("FuelLevelLiters").std().alias("std_fuel_level"),
                ]
            )
            .sort("Shift")
        )

        return {"shift_analysis": shift_stats.to_dicts()}

    def get_equipment_ranking(self, top_n: int = 20) -> Dict[str, Any]:
        """
        Ranking de equipos por cantidad de registros de recarga.

        Args:
            top_n: Número de equipos a retornar
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        ranking = (
            self.supply_df.group_by(["Equipment", "TruckFleet"])
            .agg(
                [
                    pl.count().alias("total_records"),
                    pl.col("FuelLevelLiters").mean().alias("avg_fuel_level"),
                ]
            )
            .sort("total_records", descending=True)
            .head(top_n)
        )

        return {
            "equipment_ranking": ranking.to_dicts(),
            "total_equipment": self.supply_df["Equipment"].n_unique(),
        }

    def get_temporal_patterns(self) -> Dict[str, Any]:
        """
        Análisis de patrones temporales: distribución por día de la semana,
        hora del día, y tendencias mensuales.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        # Distribución por día de la semana
        weekday_dist = (
            self.supply_df.group_by("Weekday")
            .agg([pl.count().alias("count")])
            .sort("Weekday")
        )

        # Distribución por hora del día
        hourly_dist = (
            self.supply_df.group_by("Hour")
            .agg([pl.count().alias("count")])
            .sort("Hour")
        )

        # Distribución mensual
        monthly_dist = (
            self.supply_df.with_columns(
                [
                    pl.col("ShiftDate").dt.month().alias("Month"),
                    pl.col("ShiftDate").dt.year().alias("Year"),
                ]
            )
            .group_by(["Year", "Month"])
            .agg([pl.count().alias("count")])
            .sort(["Year", "Month"])
        )

        return {
            "weekday_distribution": weekday_dist.to_dicts(),
            "hourly_distribution": hourly_dist.to_dicts(),
            "monthly_distribution": monthly_dist.to_dicts(),
        }

    def close(self):
        """Cerrar conexión con ClickHouse"""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
                logging.info("✅ Conexión cerrada")
            except Exception as e:
                logging.warning(f"⚠️ Error cerrando conexión: {e}")


if __name__ == "__main__":
    eda = FuelSupplyEDA(truck_id="T-210")
    eda.run()
    df = eda.correlate_supply_events()
    print(df.head(10))
