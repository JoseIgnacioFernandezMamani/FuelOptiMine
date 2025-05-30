# load/implementations/clickhouse_loader.py
import logging
import clickhouse_connect
import polars as pl
from typing import Type, Optional
from pydantic import BaseModel
from load.utils.pydantic_to_clickhouse import pydantic_to_clickhouse
from datetime import datetime

class ClickHouseLoader:
    def __init__(self, host: str, port: int, username: str, password: str, 
                 database: str = 'default', **kwargs):
        self.client = clickhouse_connect.get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            **kwargs
        )
        self.database = database
        logging.info(f"Conexión establecida con {database}@{host}:{port}")

    def create_table_from_schema(self, table_name: str, schema_model: Type[BaseModel], 
                                engine: str = 'MergeTree', order_by: Optional[str] = None):
        """Crea tabla usando definición de esquema Pydantic con optimizaciones CH"""
        ch_columns = pydantic_to_clickhouse(schema_model)
        
        # Configuración de optimización para datos de series temporales
        if order_by is None:
            # Detectar automáticamente columnas temporales
            time_cols = [name for name, field in schema_model.model_fields.items() 
                         if field.annotation in [datetime, pl.Datetime]]
            order_by = time_cols[0] if time_cols else "tuple()"
        
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {ch_columns}
        ) ENGINE = {engine}
        ORDER BY {order_by