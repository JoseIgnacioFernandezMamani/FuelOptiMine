from etl_core.load.utils.config import CH_CONFIG, DATASET_CONFIG, create_client
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
        self.client = None
        self.params = {**CH_CONFIG, **params}
        self.metrics = {
            "total_rows": 0,
            "last_duration": 0.0,
            "avg_rps": 0.0,
            "datasets_loaded": {},
        }
        self.logger = logging.getLogger("ClickHouseLoader")
        # Umbrales de memoria en MB (bajo, medio, alto)
        self.mem_thresholds = [128, 512, 1024]
        # Mapeo de nombres de dataset para consistencia
        self.DATASET_NAME_MAPPING = {
            "sensor": "sensor"  # Corrige inconsistencia de nombres
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Connects to the ClickHouse client using configured parameters"""
        self.client = create_client(params=self.params, logger=self.logger)

    def load(self, table_name: str, df: pl.DataFrame):
        """Optimized DataFrame loading into specific table"""
        if df.is_empty():
            self.logger.warning(f"Intento de carga vacía en tabla {table_name}")
            return

        start_time = time.monotonic()

        # 1. Type optimization
        df = self.optimize_types(df)

        # 2. Dynamic batch size calculation
        batch_size = self.calculate_batch_size(df)

        # 3. Load via Arrow streaming
        try:
            self.client.insert_arrow(
                table=table_name,
                arrow_table=df.to_arrow(),
                settings={"max_insert_block_size": batch_size},
            )
        except Exception as e:
            self.logger.exception(f"Error cargando datos en {table_name}: {str(e)}")
            raise

        # 5. Metrics update
        duration = time.monotonic() - start_time
        self.update_metrics(len(df), duration, table_name)

        self.logger.info(
            f"Cargados {len(df)} registros en {table_name} en {duration:.2f}s"
        )
        return True

    def load_dataset(self, dataset_name: str, df: pl.DataFrame):
        """Loads a DataFrame into its corresponding table using the dataset name"""
        # Normalize dataset name
        normalized_name = self.DATASET_NAME_MAPPING.get(dataset_name, dataset_name)

        if normalized_name not in DATASET_CONFIG:
            self.logger.error(
                f"Dataset '{dataset_name}' (normalizado: '{normalized_name}') no configurado"
            )
            raise ValueError(f"Dataset no configurado: {normalized_name}")

        table_name = DATASET_CONFIG[normalized_name]["table_name"]
        return self.load(table_name, df)

    def optimize_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimizes data types for ClickHouse"""
        conversions = []
        for col, dtype in df.schema.items():
            if dtype == pl.Boolean:
                conversions.append(pl.col(col).cast(pl.UInt8))
            elif dtype == pl.Categorical:
                conversions.append(pl.col(col).cast(pl.Utf8))
            elif dtype == pl.Datetime:
                # Convertir a UTC si no está en zona horaria
                conversions.append(pl.col(col).dt.convert_time_zone("UTC"))

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
                return min(int(rows_per_mb * 500), 500000)  # Lotes grandes
            elif free_mem > self.mem_thresholds[1]:
                return min(int(rows_per_mb * 100), 100000)  # Lotes medianos
            else:
                return min(int(rows_per_mb * 10), 10000)  # Lotes pequeños
        except Exception as e:
            self.logger.warning(
                f"Error cálculo batch size: {str(e)} - Usando default 10k"
            )
            return 10000

    def update_metrics(self, rows: int, duration: float, table_name: str):
        """Updates performance statistics"""
        self.metrics["total_rows"] += rows
        self.metrics["last_duration"] = duration

        # Register by dataset
        dataset_name = next(
            (
                name
                for name, config in DATASET_CONFIG.items()
                if config["table_name"] == table_name
            ),
            "unknown",
        )
        self.metrics["datasets_loaded"][dataset_name] = (
            self.metrics["datasets_loaded"].get(dataset_name, 0) + rows
        )

        if duration > 0:
            rps = rows / duration
            # Media móvil exponencial
            self.metrics["avg_rps"] = 0.8 * self.metrics["avg_rps"] + 0.2 * rps

    def close(self):
        if self.client:
            self.client.close()
            self.logger.info("Conexión a ClickHouse cerrada")

    def get_metrics(self) -> Dict[str, Any]:
        """Returns accumulated performance metrics"""
        return self.metrics
