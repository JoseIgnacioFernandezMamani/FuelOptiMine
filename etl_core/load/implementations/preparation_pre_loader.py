# load_data.py
import clickhouse_connect
import polars as pl
from datetime import datetime
import os

# Configuración de ClickHouse
CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
CH_USER = os.getenv("CLICKHOUSE_USER", "default")
CH_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "password")
CH_DATABASE = os.getenv("CLICKHOUSE_DB", "fuel_optimine")


# 1. Conectar a ClickHouse y crear la tabla
def create_table():
    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DATABASE,
    )

    # Leer y ejecutar el DDL
    with open("create_lstm_fuel.sql", "r") as f:
        ddl_query = f.read()

    client.command(ddl_query)
    print("✅ Tabla lstm_fuel creada exitosamente")


# 2. Cargar y transformar datos con Polars
def load_data():
    # Cargar datos de sensores (con nuevas columnas)
    sensor_df = pl.read_csv("sensor_data.csv").with_columns(
        SlopePercent=pl.lit(None).cast(pl.Float64),
        DistanceTraveled=pl.lit(None).cast(pl.Float64),
    )

    # Cargar datos de combustible
    fuel_df = pl.read_csv("fuel_supply_data.csv").with_columns(
        # Añadir campos faltantes como nulos
        TimeStamp=pl.lit(None).cast(pl.Datetime),
        SlopePercent=pl.lit(None).cast(pl.Float64),
        DistanceTraveled=pl.lit(None).cast(pl.Float64),
    )

    # Cargar otros datasets (time_mode, cycle) de manera similar...

    # Combinar todos los DataFrames
    combined_df = pl.concat(
        [
            sensor_df,
            fuel_df,
            # ... agregar otros datasets aquí ...
        ]
    )

    # Escribir en ClickHouse
    combined_df.write_database(
        table_name="lstm_fuel",
        connection=f"clickhouse://{CH_USER}:{CH_PASSWORD}@{CH_HOST}:{CH_PORT}/{CH_DATABASE}",
        if_table_exists="append",  # append/replace
        engine="clickhouse",
    )
    print(f"✅ Datos cargados: {combined_df.height} filas insertadas")

    # En load_data.py
    def optimized_load():
        # Leer datos en streaming (para archivos grandes)
        sensor_df = pl.scan_csv("sensor_data.parquet").with_columns(
            SlopePercent=pl.lit(None).cast(pl.Float64),
            DistanceTraveled=pl.lit(None).cast(pl.Float64),
        )

        # Convertir a formato eficiente para ClickHouse
        combined_df = pl.concat(
            [
                sensor_df,
                # ... otros datasets ...
            ]
        ).collect(streaming=True)

        # Insertar por lotes
        batch_size = 100_000
        for i in range(0, combined_df.height, batch_size):
            batch = combined_df.slice(i, batch_size)
            batch.write_database(
                table_name="lstm_fuel",
                connection=f"clickhouse://{CH_USER}:{CH_PASSWORD}@{CH_HOST}:{CH_PORT}/{CH_DATABASE}",
                if_table_exists="append",
                engine="clickhouse",
            )
            print(f"✅ Lote {i//batch_size} insertado: {batch.height} filas")


if __name__ == "__main__":
    create_table()
    load_data()
