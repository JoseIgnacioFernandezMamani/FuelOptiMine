import importlib
import logging
from typing import Type, get_origin, get_args, Union, Any
import typing
from pydantic import BaseModel
import sys
from types import ModuleType

# Mapeo completo de tipos
TYPE_MAPPING: dict[str, str] = {
    "int": "Int64",
    "float": "Float64",
    "str": "String",
    "bool": "UInt8",
    "datetime": "DateTime",
    "date": "Date",
    "UUID": "UUID",
    "Enum": "Enum8",
}


def get_base_type_name(annotation: Any) -> str:
    """Obtiene el nombre base del tipo, manejando uniones y casos complejos"""
    # Caso especial para typing.Any
    if annotation is typing.Any:
        return "Any"

    # Manejar tipos de unión (Union o |)
    origin: Any = get_origin(annotation)
    if origin is Union:
        args: tuple[Any, ...] = get_args(annotation)
        # Filtrar None y obtener el primer tipo no nulo
        non_none_types: list[Any] = [t for t in args if t not in (None, type(None))]
        if non_none_types:
            return get_base_type_name(non_none_types[0])
        return "Any"  # Fallback

    # Manejar tipos estándar con nombre
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Manejar representaciones de cadena complejas
    type_str: str = str(annotation)
    if "[" in type_str:  # Para tipos genéricos
        return type_str.split("[")[0]
    if "." in type_str:  # Para tipos de módulos
        return type_str.split(".")[-1].rstrip("'>")

    # Fallback a representación de cadena limpia
    return type_str.replace("typing.", "").replace("class ", "").replace("'", "")


def import_schema_class(schema_path: str) -> Type[BaseModel]:
    """Importa dinámicamente un esquema Pydantic con mejor manejo de errores"""
    try:
        module_name, class_name = schema_path.rsplit(".", 1)

        print(f"🔍 Intentando importar: {module_name}.{class_name}")

        try:
            module: ModuleType = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            print(f"❌ Módulo no encontrado: {module_name}")
            print("💡 Directorios en sys.path:")
            for path in sys.path:
                print(f" - {path}")
            raise

        if not hasattr(module, class_name):
            available: list[str] = [
                attr for attr in dir(module) if not attr.startswith("__")
            ]
            print(f"❌ Clase no encontrada: {class_name} en {module_name}")
            print(f"💡 Atributos disponibles: {', '.join(available)}")
            raise AttributeError(
                f"El módulo '{module_name}' no tiene atributo '{class_name}'"
            )

        schema_class: Any = getattr(module, class_name)
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
    columns: list[str] = []
    for field_name, field in model.model_fields.items():
        # Manejar tipos complejos y Nullable
        annotation = field.annotation

        # Manejar tipos opcionales (Nullable)
        is_optional: bool = not field.is_required()
        base_type: str = get_base_type_name(annotation)

        # Mapear tipo base
        ch_type: str = TYPE_MAPPING.get(base_type, "String")

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
        comment: str = f" COMMENT '{field.description}'" if field.description else ""
        columns.append(f"    {field_name} {ch_type}{comment}")

    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
{',\n'.join(columns)}
    ) ENGINE = {engine}
    ORDER BY {order_by}
    SETTINGS index_granularity=8192
    """
