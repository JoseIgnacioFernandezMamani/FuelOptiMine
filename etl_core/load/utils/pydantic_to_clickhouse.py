# load/utils/pydantic_to_clickhouse.py
from pydantic import BaseModel
from typing import Type


def pydantic_to_clickhouse(model: Type[BaseModel]) -> str:
    """Convierte un modelo Pydantic a definición de tabla ClickHouse"""
    type_mapping = {
        "date": "Date",
        "datetime": "DateTime",
        "str": "String",
        "float": "Float64",
        "int": "Int32",
        "bool": "UInt8",
    }

    columns = []
    for field_name, field in model.model_fields.items():
        py_type = field.annotation.__name__.lower()
        ch_type = type_mapping.get(py_type, "String")

        # Manejar campos requeridos
        nullability = "" if field.is_required() else "Nullable"
        if nullability:
            ch_type = f"{nullability}({ch_type})"

        columns.append(f"{field_name} {ch_type}")

    return ", ".join(columns)
