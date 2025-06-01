from etl_core.load.utils.config import CH_CONFIG, DATASET_CONFIG
from etl_core.load.utils.schema_handler import (
    import_schema_class,
    pydantic_to_clickhouse,
)
import clickhouse_connect
import logging
import time


class ClickHouseInitializer:
    """Inicializador completo de base de datos y tablas en ClickHouse"""

    def __init__(self, **params):
        self.client = None
        self.params = {**CH_CONFIG, **params}
        self.logger = logging.getLogger("ClickHouseInitializer")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        """Establece conexión con reintento exponencial"""
        for attempt in range(3):
            try:
                self.client = clickhouse_connect.get_client(**self.params)
                self.client.command("SELECT 1")  # Test de conexión
                self.logger.info("Conexión exitosa a ClickHouse")
                return
            except Exception as e:
                wait_time = 2**attempt
                self.logger.warning(
                    f"Intento {attempt+1} fallido. Reintentando en {wait_time}s..."
                )
                time.sleep(wait_time)
        raise ConnectionError("No se pudo conectar a ClickHouse después de 3 intentos")

    def create_database(self):
        """Crea la base de datos si no existe"""
        db_name = self.params["database"]
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        self.logger.info(f"Base de datos '{db_name}' creada/verificada")

    def create_table(self, dataset_name: str):
        """Crea una tabla para un dataset específico"""
        if dataset_name not in DATASET_CONFIG:
            self.logger.error(
                f"Dataset '{dataset_name}' no encontrado en configuración"
            )
            raise ValueError(f"Dataset no configurado: {dataset_name}")

        config = DATASET_CONFIG[dataset_name]
        try:
            schema_class = import_schema_class(config["schema_path"])
            ddl = pydantic_to_clickhouse(
                model=schema_class,
                table_name=config["table_name"],
                order_by=config["order_by"],
                engine=config["engine"],
            )
            self.client.command(ddl)
            self.logger.info(
                f"Tabla '{config['table_name']}' creada/verificada para dataset '{dataset_name}'"
            )
        except Exception as e:
            self.logger.exception(f"Error creando tabla para {dataset_name}: {str(e)}")
            raise

    def initialize_database(self, datasets=None):
        """
        Inicializa toda la base de datos o datasets específicos

        Args:
            datasets: Lista de datasets a inicializar (None = todos)
        """
        self.create_database()

        # Determinar qué datasets inicializar
        target_datasets = datasets if datasets else DATASET_CONFIG.keys()

        for dataset in target_datasets:
            self.create_table(dataset)

        self.logger.info("Base de datos inicializada exitosamente")

    def close(self):
        if self.client:
            self.client.close()
            self.logger.info("Conexión a ClickHouse cerrada")
