from etl_core.load.utils.config import CH_CONFIG, DATASET_CONFIG, create_client
from etl_core.load.utils.schema_handler import (
    import_schema_class,
    pydantic_to_clickhouse,
)
import logging


class ClickHouseInitializer:
    """Complete initializer for database and tables in ClickHouse"""

    def __init__(self, **params):
        self.client = None
        self.params = {**CH_CONFIG, **params}
        self.logger = logging.getLogger("ClickHouseInitializer")

    def __enter__(self):
        """Executed when entering the with block"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Executed when exiting the with block"""
        self.close()

    def connect(self):
        """Usa la utilidad reutilizable"""
        self.client = create_client(params=self.params, logger=self.logger)

    def create_database(self):
        """Creates the database if it does not exist"""
        db_name = self.params["database"]
        self.client.command(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        self.logger.info(f"Base de datos '{db_name}' creada/verificada")

    def create_table(self, dataset_name: str):
        """Creates a table for a specific dataset"""
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
        Initializes the entire database or specific datasets

        Args:
            datasets: List of datasets to initialize (None = all)
        """
        self.create_database()

        # Determinar qué datasets inicializar
        target_datasets = datasets if datasets else DATASET_CONFIG.keys()

        for dataset in target_datasets:
            self.create_table(dataset)

        self.logger.info("Base de datos inicializada exitosamente")

    def close(self):
        """
        Closes the connection to ClickHouse
        """
        if self.client:
            try:
                self.client.close()
                self.logger.info("Conexión a ClickHouse cerrada")
            except Exception as e:
                self.logger.error(f"Error cerrando conexión: {str(e)}")
            finally:
                self.client = None
