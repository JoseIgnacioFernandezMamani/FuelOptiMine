from attr.filters import exclude
from etl_core.load.utils.config import CH_CONFIG, create_client
import time
from datetime import datetime
import os
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

    @staticmethod
    def unify_dataframes(
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_cycle: pl.DataFrame,
        max_tolerance_days: int = 365,
    ) -> pl.DataFrame:
        if len(df_sensor) == 0 or len(df_time_model) == 0 or len(df_cycle) == 0:
            raise ValueError("One or more required dataframes are empty")

        # ordenar dataframes
        df_sensor = df_sensor.sort(["TruckFleet", "Equipment", "TimeStamp"])
        df_time_model = df_time_model.sort(["TruckFleet", "Equipment", "TimeStamp"])
        df_cycle = df_cycle.sort(["TruckFleet", "Equipment", "TimeStamp"])

        # Unir sensor con time_model
        df_unified = df_sensor.join_asof(
            df_time_model,
            on="TimeStamp",
            strategy="forward",
            suffix="_tm",
            tolerance=f"{max_tolerance_days}d",
            coalesce=False,
            allow_parallel=True,
        )
        # unir con cycle
        df_unified = df_unified.join_asof(
            df_cycle,
            on="TimeStamp",
            strategy="forward",
            suffix="_cycle",
            tolerance=f"{max_tolerance_days}d",
            coalesce=False,
            allow_parallel=True,
        )

        ## obtener los timemodelid que no aparecen en unified_df
        df_missing_tm = df_time_model.join(
            df_unified.select("TimeModelId").unique(),
            on="TimeModelId",
            how="anti",
        )
        df_missing_cycle = df_cycle.join(
            df_unified.select("CycleId", "StageSequence").unique(),
            on=["CycleId", "StageSequence"],
            how="anti",
        )

        # Obtener columnas originales
        time_model_columns = df_time_model.columns
        cycle_columns = df_cycle.columns

        # Crear diccionarios de renombrado dinámicamente SOLO para las columnas que aparecen duplicadas
        # Las columnas únicas (como CycleId, TimeModelId, StageSequence, etc.) NO se renombran

        # Encontrar columnas que ya tienen sufijo en df_unified (fueron renombradas automáticamente)
        cols_with_tm_suffix = [col for col in df_unified.columns if col.endswith("_tm")]
        cols_with_cycle_suffix = [
            col for col in df_unified.columns if col.endswith("_cycle")
        ]

        # Crear mapeo solo para las columnas que necesitan renombrado
        # (las que fueron renombradas automáticamente en el join)
        original_tm_cols = [col.replace("_tm", "") for col in cols_with_tm_suffix]
        original_cycle_cols = [
            col.replace("_cycle", "") for col in cols_with_cycle_suffix
        ]

        rename_map_tm = {col: f"{col}_tm" for col in original_tm_cols}
        rename_map_cycle = {col: f"{col}_cycle" for col in original_cycle_cols}

        df_missing_tm_renamed = df_missing_tm.rename(rename_map_tm)
        df_missing_cycle_renamed = df_missing_cycle.rename(rename_map_cycle)

        # columnas que tiene el unified DESPUÉS de todos los joins
        unified_cols = df_unified.columns

        # asegurar que df_missing_tm_renamed tenga todas las columnas de unified
        missing_cols_tm = [
            col for col in unified_cols if col not in df_missing_tm_renamed.columns
        ]
        if missing_cols_tm:
            df_missing_tm_renamed = df_missing_tm_renamed.with_columns(
                [pl.lit(None).alias(col) for col in missing_cols_tm]
            )

        # asegurar que df_missing_cycle_renamed tenga todas las columnas de unified
        missing_cols_cycle = [
            col for col in unified_cols if col not in df_missing_cycle_renamed.columns
        ]
        if missing_cols_cycle:
            df_missing_cycle_renamed = df_missing_cycle_renamed.with_columns(
                [pl.lit(None).alias(col) for col in missing_cols_cycle]
            )

        # reordenar columnas para que el orden coincida exactamente
        df_missing_tm_renamed = df_missing_tm_renamed.select(unified_cols)
        df_missing_cycle_renamed = df_missing_cycle_renamed.select(unified_cols)

        # Concatenar todos los dataframes
        df_unified = pl.concat(
            [df_unified, df_missing_tm_renamed, df_missing_cycle_renamed],
            how="vertical",
        )

        # Crear columna temporal para ordenamiento y ordenar
        df_unified = df_unified.with_columns(
            pl.coalesce(
                [pl.col("TimeStamp"), pl.col("TimeStamp_tm"), pl.col("TimeStamp_cycle")]
            ).alias("SortTimestamp")
        ).sort("SortTimestamp")

        # obtener todas las columnas
        exclude_columns = ["TimeModelId", "CycleId"]
        columns = df_unified.columns
        columns_filter = [col for col in columns if col not in exclude_columns]

        # Eliminar columnas
        df_unified = df_unified.with_columns(
            pl.when(pl.col(col) == pl.col(col).shift(-1))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in columns_filter
        )

        return df_unified

    def load(
        self,
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ):
        """Optimized DataFrame loading into specific table"""
        try:
            # Unificar los dataframes
            df_unified = self.unify_dataframes(df_sensor, df_time_model, df_cycle)

            if df_unified.is_empty():
                self.logger.warning(
                    f"Intento de carga vacía en tabla {self.TARGET_TABLE}"
                )
                return False

            start_time = time.monotonic()

            # 1. Type optimization
            df_unified = self.optimize_types(df_unified)

            # 2. Dynamic batch size calculation
            batch_size = self.calculate_batch_size(df_unified)

            # 3. Load via Arrow streaming
            self.client.insert_arrow(
                table=self.TARGET_TABLE,
                arrow_table=df_unified.to_arrow(),
                settings={"max_insert_block_size": batch_size},
            )

            # 4. Metrics update
            duration = time.monotonic() - start_time
            self.update_metrics(len(df_unified), duration)

            self.logger.info(
                f"Cargados {len(df_unified)} registros en {self.TARGET_TABLE} en {duration:.2f}s"
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"Error cargando datos en {self.TARGET_TABLE}: {str(e)}"
            )
            raise

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
            try:
                self.client.close()
                self.logger.info("Conexión a ClickHouse cerrada")
            except Exception as e:
                self.logger.warning(f"Error cerrando conexión: {str(e)}")
            finally:
                self.client = None

    def get_metrics(self) -> Dict[str, Any]:
        """Returns accumulated performance metrics"""
        return self.metrics
