import os
import logging
import time
import clickhouse_connect
from clickhouse_connect.driver.client import Client
from typing import Any

# Configuración de conexión
CH_CONFIG: dict[str, bool | int | str] = {
    "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
    "port": int(os.getenv("CLICKHOUSE_NATIVE_PORT", 8123)),
    "username": os.getenv("CLICKHOUSE_USER", "default"),
    "password": os.getenv("CLICKHOUSE_PASSWORD", "password"),
    "database": os.getenv("CLICKHOUSE_DB", "fuel_optimine"),
    "compress": True,
    "send_receive_timeout": 300,
}


def create_client(params, logger=None) -> Client:
    """
    Creates a ClickHouse client with exponential backoff retry
    Returns: clickhouse_connect.driver.client.Client
    """
    logger = logger or logging.getLogger("ClickHouseConnection")

    for attempt in range(3):
        try:
            client: Client = clickhouse_connect.get_client(**params)
            client.command("SELECT 1")  # Connection test
            logger.info("Conexión exitosa a ClickHouse")
            return client
        except Exception as e:
            wait_time: Any = 2**attempt
            logger.warning(
                f"Intento {attempt+1} fallido. Reintentando en {wait_time}s... Error: {str(e)}"
            )
            time.sleep(wait_time)

    raise ConnectionError("No se pudo conectar a ClickHouse después de 3 intentos")
