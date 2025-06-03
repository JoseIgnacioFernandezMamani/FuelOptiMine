from etl_core.load.utils.config import CH_CONFIG, DATASET_CONFIG
import clickhouse_connect
import time
import polars as pl
import psutil
import logging
from typing import Dict, Any


class ClickHouseLoader:
    """
    Loader optimizado para carga masiva en ClickHouse con:
    - Protocolo nativo
    - Streaming Arrow
    - Batch sizing automático
    - Manejo de esquemas y tipos
    - Métricas de rendimiento
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
        """Conexión con reintento exponencial"""
        for attempt in range(3):
            try:
                self.client = clickhouse_connect.get_client(**self.params)
                self.client.command("SELECT 1")  # Test de conexión
                self.logger.info("Conexión exitosa a ClickHouse")
                return
            except Exception as e:
                wait_time = 2**attempt
                self.logger.warning(
                    f"Intento {attempt+1} fallido. Reintentando en {wait_time}s... Error: {str(e)}"
                )
                time.sleep(wait_time)
        raise ConnectionError("No se pudo conectar a ClickHouse después de 3 intentos")

    def load(self, table_name: str, df: pl.DataFrame):
        """Carga optimizada de DataFrame en tabla específica"""
        if df.is_empty():
            self.logger.warning(f"Intento de carga vacía en tabla {table_name}")
            return

        start_time = time.monotonic()

        # 1. Optimización de tipos
        df = self.optimize_types(df)

        # 2. Cálculo de batch size dinámico
        batch_size = self.calculate_batch_size(df)

        # 3. Carga por streaming Arrow
        try:
            self.client.insert_arrow(
                table=table_name,
                arrow_table=df.to_arrow(),
                settings={"max_insert_block_size": batch_size},
            )
        except Exception as e:
            self.logger.exception(f"Error cargando datos en {table_name}: {str(e)}")
            raise

        # 5. Actualización de métricas
        duration = time.monotonic() - start_time
        self.update_metrics(len(df), duration, table_name)

        self.logger.info(
            f"Cargados {len(df)} registros en {table_name} en {duration:.2f}s"
        )
        return True

    def load_dataset(self, dataset_name: str, df: pl.DataFrame):
        """Carga un DataFrame en su tabla correspondiente usando nombre de dataset"""
        # Normalizar nombre del dataset
        normalized_name = self.DATASET_NAME_MAPPING.get(dataset_name, dataset_name)

        if normalized_name not in DATASET_CONFIG:
            self.logger.error(
                f"Dataset '{dataset_name}' (normalizado: '{normalized_name}') no configurado"
            )
            raise ValueError(f"Dataset no configurado: {normalized_name}")

        table_name = DATASET_CONFIG[normalized_name]["table_name"]
        return self.load(table_name, df)

    def optimize_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Optimiza tipos de datos para ClickHouse"""
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
        """Calcula tamaño de lote basado en memoria libre y tamaño de datos"""
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
        """Actualiza estadísticas de rendimiento"""
        self.metrics["total_rows"] += rows
        self.metrics["last_duration"] = duration

        # Registrar por dataset
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
        """Devuelve métricas de rendimiento acumuladas"""
        return self.metrics
