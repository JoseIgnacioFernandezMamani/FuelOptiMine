from etl_core.load.utils.config import CH_CONFIG, create_client
from etl_core.load.implementations.data_unifier import DataUnifier
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
    - Integration with DataUnifier
    """

    def __init__(self, max_tolerance_days: int = 365, **params):
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
        self.XGBOOST_TABLE = "xgboost_fuel"
        self.FUEL_SUPPLY_TABLE = "fuel_supply"
        # Tolerancia máxima para unificación de datos
        self.max_tolerance_days = max_tolerance_days

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Connects to the ClickHouse client using configured parameters"""
        self.client = create_client(params=self.params, logger=self.logger)

    def load_unified_data(
        self,
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ):
        """Optimized DataFrame loading into specific table using DataUnifier"""
        try:
            self.logger.info("Iniciando unificación de datos con DataUnifier")
            # Unificar los dataframes
            unifier = DataUnifier(max_tolerance_days=365)
            unified_df = unifier.unify(df_sensor, df_time_model, df_cycle)

            if unified_df.is_empty():
                self.logger.warning(
                    f"Intento de carga vacía en tabla {self.XGBOOST_TABLE}"
                )
                return False

            self.logger.info(f"Datos unificados: {len(unified_df)} registros")
            start_time = time.monotonic()

            # 1. Type optimization
            unified_df = self.optimize_types(unified_df)

            # 2. Dynamic batch size calculation
            batch_size = self.calculate_batch_size(unified_df)
            self.logger.info(f"Batch size calculado: {batch_size}")

            # 3. Load via Arrow streaming
            full_table_name = f"{self.params['database']}.{self.XGBOOST_TABLE}"
            self.client.insert_arrow(
                table=full_table_name,
                arrow_table=unified_df.to_arrow(),
                settings={"max_insert_block_size": batch_size},
            )

            # 4. Metrics update
            duration = time.monotonic() - start_time
            self.update_metrics(len(unified_df), duration)

            self.logger.info(
                f"Cargados {len(unified_df)} registros en {full_table_name} en {duration:.2f}s"
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"Error cargando datos en {self.XGBOOST_TABLE}: {str(e)}"
            )
            raise

    def load_fuel_supply_data(self, df: pl.DataFrame):
        """loader for fuel_supply table"""
        try:
            if df.is_empty():
                self.logger.warning(f"Intento de carga vacía en tabla fuel_supply")
                return False

            start_time = time.monotonic()

            # 1. Type optimization
            df = self.optimize_types(df)

            # 2. Dynamic batch size calculation
            batch_size = self.calculate_batch_size(df)
            self.logger.info(f"Batch size calculado: {batch_size}")

            # 3. Load via Arrow streaming
            full_table_name = f"{self.params['database']}.{self.FUEL_SUPPLY_TABLE}"
            self.client.insert_arrow(
                table=full_table_name,
                arrow_table=df.to_arrow(),
                settings={"max_insert_block_size": batch_size},
            )

            # 4. Metrics update
            duration = time.monotonic() - start_time
            self.update_metrics(len(df), duration)

            self.logger.info(
                f"Cargados {len(df)} registros en {full_table_name} en {duration:.2f}s"
            )
            return True

        except Exception as e:
            self.logger.exception(f"Error cargando datos en fuel_supply: {str(e)}")
            raise

    def optimize_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimizes data types for ClickHouse with XGBoost table considerations"""
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
                conversions.append(
                    pl.col(col).dt.convert_time_zone("UTC").dt.cast_time_unit("ms")
                )
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
            # Media móvil exponencial para suavizar métricas
            self.metrics["avg_rps"] = 0.8 * self.metrics["avg_rps"] + 0.2 * rps

    def get_table_info(self) -> Dict[str, Any]:
        """Returns information about the target table"""
        try:
            full_table_name = f"{self.params['database']}.{self.XGBOOST_TABLE}"

            # Obtener información de la tabla
            table_info = self.client.command(f"DESCRIBE TABLE {full_table_name}")
            row_count = self.client.command(f"SELECT COUNT(*) FROM {full_table_name}")

            return {
                "table_name": full_table_name,
                "columns": len(table_info),
                "total_rows": row_count,
                "structure": table_info,
            }
        except Exception as e:
            self.logger.error(f"Error obteniendo info de tabla: {str(e)}")
            return {}

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
        return {
            **self.metrics,
            "target_table": self.XGBOOST_TABLE,
            "max_tolerance_days": self.max_tolerance_days,
        }
