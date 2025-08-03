from etl_core.load import ClickHouseLoader
from etl_core.transform.implementation import (
    SensorTransformer,
    CycleTransformer,
    FuelSupplyTransformer,
    TimeModelTransformer,
)


def run_etl_pipeline():
    with ClickHouseLoader() as loader:
        # Procesar y cargar datasets individuales
        df_sensor = SensorTransformer().run_transform(raw_sensor_data)
        loader.load_dataset("sensor", df_sensor)

        # O cargar múltiples datasets
        transformers = {
            "cycle": CycleTransformer(),
            "fuel_supply": FuelSupplyTransformer(),
            "time_mode": TimeModelTransformer(),
        }

        for dataset_name, transformer in transformers.items():
            df = transformer.run_transform(raw_data[dataset_name])
            loader.load_dataset(dataset_name, df)

        # Mostrar métricas
        print("\n📊 Resumen de carga:")
        print(f"Total filas: {loader.metrics['total_rows']}")
        print(f"Rendimiento promedio: {loader.metrics['avg_rps']:.0f} filas/seg")
        print("Detalle por dataset:")
        for dataset, rows in loader.metrics["datasets_loaded"].items():
            print(f"- {dataset}: {rows} filas")
