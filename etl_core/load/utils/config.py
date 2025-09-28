import os
import logging
import time
import clickhouse_connect
from clickhouse_connect.driver.client import Client
from typing import Any

# Configuración de conexión
CH_CONFIG: dict[str, Any] = {
    "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
    "port": int(os.getenv("CLICKHOUSE_NATIVE_PORT", 8123)),
    "username": os.getenv("CLICKHOUSE_USER", "default"),
    "password": os.getenv("CLICKHOUSE_PASSWORD", "password"),
    "database": os.getenv("CLICKHOUSE_DB", "fuel_optimine"),
    "compress": True,
}


def create_client(params, logger=None) -> Client:
    """
    Creates a ClickHouse client with exponential backoff retry

    Args:
        params: Connection parameters (defaults to CH_CONFIG if None)
        logger: Logger instance (creates new one if None)

    Returns:
        clickhouse_connect.driver.client.Client

    Raises:
        ConnectionError: If connection fails after 3 attempts
    """
    if params is None:
        params = CH_CONFIG

    logger = logger or logging.getLogger("ClickHouseConnection")

    for attempt in range(3):
        try:
            client: Client = clickhouse_connect.get_client(**params)
            client.command("SELECT 1", settings={"max_execution_time": 5})
            logger.info(
                f"Conexión exitosa a ClickHouse en {params['host']}:{params['port']}"
            )
            return client
        except Exception as e:
            wait_time: Any = 2**attempt
            logger.warning(
                f"Intento {attempt+1} fallido. Reintentando en {wait_time}s... Error: {str(e)}"
            )
            if attempt < 2:
                time.sleep(wait_time)

    raise ConnectionError(
        f"No se pudo conectar a ClickHouse en {params['host']}:{params['port']} "
        f"después de 3 intentos. Verifique la conectividad y credenciales."
    )
