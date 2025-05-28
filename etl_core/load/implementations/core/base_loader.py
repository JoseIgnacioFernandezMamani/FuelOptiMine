from clickhouse_driver import connect
from typing import Type, Dict, Any, List
from pydantic import BaseModel
import logging


class BaseClickHouseLoader:
    """Cargador base que usa esquemas Pydantic"""

    def __init__(
        self,
        schema: Type[BaseModel],
        table_name: str,
        host="localhost",
        port=9000,
        user="default",
        password="password",
        database="default",
    ):
        self.schema = schema
        self.table_name = table_name
        self.config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "compression": "lz4",
            "settings": {"max_block_size": 100000, "use_numpy": True},
        }
        self._ensure_table_exists()

    def _pydantic_to_clickhouse_type(self, field_type) -> str:
        """Mapea tipos Pydantic a tipos ClickHouse"""
        type_mapping = {
            "date": "Date",
            "datetime": "DateTime64(6)",
            "float": "Float64",
            "str": "String",
            "int": "Int64",
            "bool": "UInt8",
        }
        return type_mapping.get(field_type.__name__.lower(), "String")

    def _generate_table_ddl(self) -> str:
        """Genera DDL desde el esquema Pydantic"""
        columns = []
        for field_name, field in self.schema.__fields__.items():
            ch_type = self._pydantic_to_clickhouse_type(field.type_)
            columns.append(f"{field_name} {ch_type}")

        return f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            {', '.join(columns)}
        ) ENGINE = MergeTree()
        ORDER BY tuple()
        SETTINGS index_granularity = 8192
        """

    def _ensure_table_exists(self):
        """Crea la tabla si no existe"""
        ddl = self._generate_table_ddl()
        with connect(**self.config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(ddl)

    def _transform_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Transformación específica por implementación"""
        return record

    def load_data(self, data: List[Dict[str, Any]], batch_size=50000):
        """Carga datos a ClickHouse"""
        if not data:
            logging.warning("Datos vacíos. No se cargará nada.")
            return 0

        # Aplicar transformaciones específicas
        transformed_data = [self._transform_record(record) for record in data]

        # Preparar inserción
        columns = list(self.schema.__fields__.keys())
        query = f"INSERT INTO {self.table_name} ({','.join(columns)}) VALUES"

        with connect(**self.config) as conn:
            with conn.cursor() as cursor:
                for i in range(0, len(transformed_data), batch_size):
                    batch = [
                        tuple(record[col] for col in columns if col in record)
                        for record in transformed_data[i : i + batch_size]
                    ]

                    cursor.executemany(query, batch)

        return len(transformed_data)
