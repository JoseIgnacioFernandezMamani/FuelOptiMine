from etl_core.load.utils.config import CH_CONFIG, create_client
import time
import polars as pl
import psutil
import logging
from typing import Dict, Any


class ClickHouseLoader:
    """
    Optimized loader for bulk loading into ClickHouse with:
    - Native protocol
    - Arrow streaming
    - Automatic batch sizing
    - Schema and type handling
    - Performance metrics
    """

    def __init__(self, **params):
        self.client: Any = None
        self.params = {**CH_CONFIG, **params}
        self.metrics = {
            "total_rows": 0,
            "last_duration": 0.0,
            "avg_rps": 0.0,
        }
        self.logger = logging.getLogger("ClickHouseLoader")
        # Umbrales de memoria en MB (bajo, medio, alto)
        self.mem_thresholds = [128, 512, 1024]
        # Tabla destino unificada
        self.TARGET_TABLE = "lstm_fuel"

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Connects to the ClickHouse client using configured parameters"""
        self.client = create_client(params=self.params, logger=self.logger)

    def unify_dataframes(
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_fuel_supply: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Unifica los dataframes de diferentes fuentes en base al timestamp de los sensores.

        Lógica de unión:
        - Para time_model y fuel_supply: última entrada <= timestamp del sensor
        - Para cycle: ciclo activo donde E_TravelingStart <= timestamp del sensor
        """
        # Convertir todas las columnas de tiempo a datetime
        df_sensor = df_sensor.with_columns(pl.col("TimeStamp").str.to_datetime())
        df_time_model = df_time_model.with_columns(
            pl.col("TimeStamp").str.to_datetime()
        )
        df_fuel_supply = df_fuel_supply.with_columns(
            pl.col("TimeStamp").str.to_datetime()
        )
        df_cycle = df_cycle.with_columns(
            pl.col("E_TravelingStart").str.to_datetime(),
            pl.col("L_UnloadingEnd").str.to_datetime(),
        )

        # 1. Unión con time_model (último registro <= timestamp del sensor)
        df_unified = df_sensor.join_asof(
            df_time_model,
            on="TimeStamp",
            by=["Equipment", "TruckFleet", "ShiftDate", "Shift"],
            strategy="backward",
            suffix="_time_model",
        )

        # 2. Unión con fuel_supply (último registro <= timestamp del sensor)
        df_unified = df_unified.join_asof(
            df_fuel_supply,
            on="TimeStamp",
            by=["Equipment", "TruckFleet", "ShiftDate", "Shift"],
            strategy="backward",
            suffix="_fuel_supply",
        )

        # 3. Unión con cycle (ciclo activo en el momento del sensor)
        df_unified = df_unified.join_asof(
            df_cycle.sort("E_TravelingStart"),
            left_on="TimeStamp",
            right_on="E_TravelingStart",
            by=["Equipment", "TruckFleet", "ShiftDate", "Shift"],
            strategy="backward",
        ).filter(
            (pl.col("TimeStamp") >= pl.col("E_TravelingStart"))
            & (pl.col("TimeStamp") <= pl.col("L_UnloadingEnd"))
        )

        return df_unified

    def load(
        self,
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_fuel_supply: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ):
        """Optimized DataFrame loading into specific table"""
        if df.is_empty():
            self.logger.warning(f"Intento de carga vacía en tabla {self.TARGET_TABLE}")
            return

        start_time = time.monotonic()

        # 1. Type optimization
        df = self.optimize_types(df)

        # 2. Dynamic batch size calculation
        batch_size = self.calculate_batch_size(df)

        # 3. Load via Arrow streaming
        try:
            self.client.insert_arrow(
                table=self.TARGET_TABLE,
                arrow_table=df.to_arrow(),
                settings={"max_insert_block_size": batch_size},
            )
        except Exception as e:
            self.logger.exception(
                f"Error cargando datos en {self.TARGET_TABLE}: {str(e)}"
            )
            raise

        # 4. Metrics update
        duration = time.monotonic() - start_time
        self.update_metrics(len(df), duration)

        self.logger.info(
            f"Cargados {len(df)} registros en {self.TARGET_TABLE} en {duration:.2f}s"
        )
        return True

    def optimize_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimizes data types for ClickHouse"""
        conversions = []
        for col, dtype in df.schema.items():
            # Convertir booleanos a UInt8
            if dtype == pl.Boolean:
                conversions.append(pl.col(col).cast(pl.UInt8))
            # Convertir categóricos a String
            elif dtype == pl.Categorical:
                conversions.append(pl.col(col).cast(pl.Utf8))
            # Convertir fechas/horas a UTC
            elif dtype == pl.Datetime:
                conversions.append(pl.col(col).dt.convert_time_zone("UTC"))
            # Convertir Float64 a Float32 para ahorrar espacio
            elif dtype == pl.Float64:
                conversions.append(pl.col(col).cast(pl.Float32))
            # Convertir Int64 a Int32 donde sea posible
            elif dtype == pl.Int64:
                conversions.append(pl.col(col).cast(pl.Int32))

        return df.with_columns(conversions) if conversions else df

    def calculate_batch_size(self, df: pl.DataFrame) -> int:
        """Calculates batch size based on free memory and data size"""
        try:
            free_mem = psutil.virtual_memory().available / (1024**2)  # MB
            df_size = df.estimated_size() / (1024**2)  # MB

            if df_size == 0 or len(df) == 0:
                return 10000

            # Filas por MB (ajuste conservador)
            rows_per_mb = len(df) / df_size

            # Selección de estrategia basada en memoria
            if free_mem > self.mem_thresholds[2]:
                return min(int(rows_per_mb * 500), 500000)
            elif free_mem > self.mem_thresholds[1]:
                return min(int(rows_per_mb * 100), 100000)
            else:
                return min(int(rows_per_mb * 10), 10000)
        except Exception as e:
            self.logger.warning(
                f"Error cálculo batch size: {str(e)} - Usando default 10k"
            )
            return 10000

    def update_metrics(self, rows: int, duration: float):
        """Updates performance statistics"""
        self.metrics["total_rows"] += rows
        self.metrics["last_duration"] = duration

        if duration > 0:
            rps: float = rows / duration
            # Media móvil exponencial
            self.metrics["avg_rps"] = 0.8 * self.metrics["avg_rps"] + 0.2 * rps

    def close(self):
        if self.client:
            self.client.close()
            self.logger.info("Conexión a ClickHouse cerrada")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns accumulated performance metrics"""
        return self.metrics
