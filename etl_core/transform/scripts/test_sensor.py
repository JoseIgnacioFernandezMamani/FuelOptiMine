from etl_core.transform.implementation.sensor.transformer import SensorTransformer
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.utils.equipment_constants import TRUCK_SPECS
import sys
import os
import polars as pl


def run_sensor_etl_pipeline() -> None:
    # Configuración de rutas y parámetros
    truck_id = "T-210"
    dataset_name = "train_data"
    data_type = "sensor"

    # Configurar rutas de importación
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    print(f"🚚 Starting ETL for truck {truck_id} - {dataset_name} (Sensor Data)")

    try:
        # 1. Data extraction
        print("\n🔍 Extracting data from CSV...")
        extractor: CSVExtractor = CSVExtractor(dataset_name, truck_id)
        raw_data: dict[str, pl.DataFrame] = extractor.load_data()

        if not raw_data or data_type not in raw_data:
            raise ValueError(
                f"Data of type '{data_type}' not found or invalid structure"
            )

        # Get Polars DataFrame directly
        df_raw: pl.DataFrame = raw_data[data_type]
        print(f"✅ Raw data loaded: {df_raw.height} records")
        print("Initial schema:", df_raw.schema)

        # 2. Data transformation
        print("\n🔄 Processing data with SensorTransformer...")
        transformer: SensorTransformer = SensorTransformer(truck_id=truck_id)
        df_clean: pl.DataFrame | None = transformer.run_transform(df_raw)

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("Transformation returned empty data")

        sample_columns: list[str] = ["TimeStamp", "Speed", "Equipment", "TruckFleet"]
        print(df_clean.select(sample_columns).head(5))

        # guardar
        output_path = os.path.join(os.getcwd(), f"{truck_id}_sensor_transformed.csv")
        df_clean.write_csv(output_path)

    except Exception as e:
        print(f"\n❌ Critical error in pipeline: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_sensor_etl_pipeline()
