from etl_core.load.utils.config import CH_CONFIG, create_client
from etl_core.utils.fuel_optimine_table import CREATE_TABLE_LSTM_FUEL
import logging
from typing import Any


class ClickHouseInitializer:
    """Complete initializer for database and tables in ClickHouse"""

    def __init__(self, **params):
        self.client: Any = None
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

    def create_table(self, table_ddl: str):
        """Creates a table using direct DDL"""
        try:
            self.client.command(table_ddl)
            self.logger.info("Tabla creada exitosamente")
        except Exception as e:
            self.logger.exception(f"Error creando tabla: {str(e)}")
            raise

    def initialize_database(self):
        """Initializes the database and creates tables"""
        self.create_database()
        self.create_table(CREATE_TABLE_LSTM_FUEL)
        self.logger.info("Base de datos inicializada exitosamente")

    def close(self):
        if self.client:
            try:
                self.client.close()
                self.logger.info("Conexión a ClickHouse cerrada")
            except Exception as e:
                self.logger.error(f"Error cerrando conexión: {str(e)}")
            finally:
                self.client = None
