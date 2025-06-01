import importlib
import logging
from typing import Type
from pydantic import BaseModel
import sys

# Mapeo completo de tipos
TYPE_MAPPING = {
    "int": "Int64",
    "float": "Float64",
    "str": "String",
    "bool": "UInt8",
    "datetime": "DateTime",
    "date": "Date",
    "UUID": "UUID",
    "Enum": "Enum8",
}


def import_schema_class(schema_path: str) -> Type[BaseModel]:
    """Importa dinámicamente un esquema Pydantic con mejor manejo de errores"""
    try:
        module_name, class_name = schema_path.rsplit(".", 1)

        print(f"🔍 Intentando importar: {module_name}.{class_name}")

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            print(f"❌ Módulo no encontrado: {module_name}")
            print("💡 Directorios en sys.path:")
            for path in sys.path:
                print(f" - {path}")
            raise

        if not hasattr(module, class_name):
            available = [attr for attr in dir(module) if not attr.startswith("__")]
            print(f"❌ Clase no encontrada: {class_name} en {module_name}")
            print(f"💡 Atributos disponibles: {', '.join(available)}")
            raise AttributeError(
                f"El módulo '{module_name}' no tiene atributo '{class_name}'"
            )

        schema_class = getattr(module, class_name)
        return schema_class

    except (ImportError, AttributeError, ValueError) as e:
        logging.exception(f"Error crítico importando esquema: {schema_path}")
        print(f"🔥 Error fatal: {str(e)}")
        raise RuntimeError(f"Fallo en importación de esquema: {schema_path}") from e


def pydantic_to_clickhouse(
    model: Type[BaseModel],
    table_name: str,
    order_by: str = "tuple()",
    engine: str = "MergeTree",
) -> str:
    """Genera DDL ClickHouse desde modelo Pydantic con manejo de tipos avanzado"""
    columns = []
    for field_name, field in model.model_fields.items():
        # Manejar tipos complejos y Nullable
        annotation = field.annotation
        type_name = (
            annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
        )

        # Manejar tipos opcionales (Nullable)
        is_optional = not field.is_required()
        base_type = type_name.replace("NoneType | ", "").replace(" | NoneType", "")

        # Mapear tipo base
        ch_type = TYPE_MAPPING.get(base_type, "String")

        # Manejar tipos especiales
        if "datetime" in base_type.lower():
            ch_type = "DateTime"
        elif "date" in base_type.lower():
            ch_type = "Date"
        elif "uuid" in base_type.lower():
            ch_type = "UUID"

        # Aplicar Nullable si es necesario
        if is_optional:
            ch_type = f"Nullable({ch_type})"

        # Agregar comentario
        comment = f" COMMENT '{field.description}'" if field.description else ""
        columns.append(f"    {field_name} {ch_type}{comment}")

    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
{',\n'.join(columns)}
    ) ENGINE = {engine}
    ORDER BY {order_by}
    SETTINGS index_granularity=8192
    """
