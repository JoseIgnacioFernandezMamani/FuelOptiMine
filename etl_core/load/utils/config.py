import os

# Configuración de conexión
CH_CONFIG = {
    "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
    "port": int(
        os.getenv("CLICKHOUSE_NATIVE_PORT", 8123)
    ),  # Puerto NATIVO para mejor rendimiento
    "username": os.getenv("CLICKHOUSE_USER", "default"),
    "password": os.getenv("CLICKHOUSE_PASSWORD", "password"),
    "database": os.getenv("CLICKHOUSE_DB", "fuel_optimine"),
    "compress": True,
    "send_receive_timeout": 300,
}
# Configuración específica por dataset
DATASET_CONFIG = {
    "sensor": {
        "table_name": "sensor_data",
        "schema_path": "etl_core.utils.sensor_schemas.SensorSchema",
        "order_by": "(Equipment, ShiftDate, TimeStamp)",
        "engine": "MergeTree",
    },
    "fuel_supply": {
        "table_name": "fuel_supply_data",
        "schema_path": "etl_core.utils.fuel_supply_schemas.FuelSupplySchema",
        "order_by": "(Veh, fin_desp)",
        "engine": "MergeTree",
    },
    "time_mode": {
        "table_name": "time_model_data",
        "schema_path": "etl_core.utils.time_model_schemas.TimeModelSchema",
        "order_by": "(Equipment, ShiftDate, TimeStamp)",
        "engine": "MergeTree",
    },
    "cycle": {
        "table_name": "cycle_data",
        "schema_path": "etl_core.utils.cycle_schemas.CycleSchema",
        "order_by": "(Equipment, ShiftDate)",
        "engine": "MergeTree",
    },
}
