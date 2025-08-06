from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.transform import (
    FuelSupplyTransformer,
    CycleTransformer,
    SensorTransformer,
    TimeModelTransformer,
)
import polars as pl


def compare_columns_for_truck(
    truck_id: str = "T-210", dataset_name: str = "train_data"
):
    data_types = {
        "time_model": ("Time Model Data", TimeModelTransformer()),
        "sensor": ("Sensor Data", SensorTransformer()),
        "fuel_supply": ("Fuel Supply Data", FuelSupplyTransformer()),  # Corregido typo
        "cycle": ("Cycle Data", CycleTransformer()),
    }

    print(f"🚚 Comparando columnas para {truck_id} (originales vs transformadas)\n")

    try:
        extractor: CSVExtractor = CSVExtractor(dataset_name, truck_id)
        raw_data: dict[str, pl.DataFrame] = extractor.load_data()
        transformer_sensor = SensorTransformer()
        transformer_cycle = CycleTransformer()
        transformer_fuel_supply = FuelSupplyTransformer()
        transformer_time_model = TimeModelTransformer()
        for data_type, (description, transformer) in data_types.items():
            if data_type not in raw_data:
                print(f"❌ No se encontraron datos de tipo '{data_type}'")
                continue

            df_raw: pl.DataFrame = raw_data[data_type]
            # Transformar los datos
            if data_type == "sensor":
                df_transformed = transformer_sensor.run_transform(df_raw)
            elif data_type == "cycle":
                df_transformed = transformer_cycle.run_transform(df_raw)
            elif data_type == "fuel_supply":
                df_transformed = transformer_fuel_supply.run_transform(df_raw)
            elif data_type == "time_model":
                df_transformed = transformer_time_model.run_transform(df_raw)
            else:
                print(f"⚠️ Tipo de dato desconocido: {data_type}")
                continue
            print(
                f"columnas transformadas {data_type} : /n \n ({description}): {df_transformed.columns} registros"
            )

    except Exception as e:
        print(f"\n🔥 ERROR GLOBAL: {str(e)}")


# Ejecutar análisis
compare_columns_for_truck("T-210")
