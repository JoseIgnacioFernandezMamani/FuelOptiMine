from etl_core.utils.cycle_schemas import CycleSchema
from etl_core.utils.sensor_schemas import SensorSchema
from etl_core.utils.time_model_schemas import TimeModelSchema


def get_pydantic_field_names(schema_class) -> list[str]:
    """Extrae los nombres de campos de un esquema Pydantic

    Args:
        schema_class: Clase de esquema Pydantic

    Returns:
        Lista de nombres de campos en el orden definido
    """
    return list(schema_class.model_fields.keys())


# Generar COLUMN_MAPPING dinámicamente desde esquemas Pydantic
COLUMN_MAPPING: dict[str, list[str]] = {
    "sensor": get_pydantic_field_names(SensorSchema),
    "time_model": get_pydantic_field_names(TimeModelSchema),
    "cycle": get_pydantic_field_names(CycleSchema),
}

SUPPORTED_FORMATS: dict[str, list[str]] = {
    "tabular": [".csv", ".tsv", ".parquet", ".feather", ".xls", ".xlsx"],
    "hierarchical": [".json", ".yaml", ".xml"],
    "binary_columnar": [".parquet", ".feather", ".orc"],
}

DATASET_TYPES: list[str] = ["train_data"]

FUEL_SUPPLY: list[str] = ["Veh", "Descripcion", "fin_desp", "volumCorregido"]
