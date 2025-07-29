from etl_core.transform.implementation.sensor.transformer import SensorTransformer
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.utils.equipment_constants import TRUCK_SPECS
import sys
import os
import polars as pl


def run_sensor_etl_for_all_trucks():
    # Crear carpeta para resultados
    output_dir = "sensor_test"
    os.makedirs(output_dir, exist_ok=True)

    # Filtrar camiones (excluyendo T-210 y T-211)
    trucks = [
        "T-210",
        "T-212",
        "T-221",
        "T-222",
        "T-223",
        "T-225",
        "T-230",
        "T-231",
        "T-232",
        "T-237",
        "T-240",
        "T-241",
        "T-243",
    ]

    print(f"🚚 Procesando {len(trucks)} camiones...")

    for truck_id in trucks:
        print(f"\n{'='*50}")
        print(f"⚙️  Iniciando ETL para {truck_id}")
        print(f"{'='*50}")

        try:
            # 1. Extracción de datos
            extractor = CSVExtractor("train_data", truck_id)
            raw_data = extractor.load_data()
            df_raw = raw_data["sensor"]

            # 2. Transformación
            transformer = SensorTransformer()
            df_clean = transformer.run_transform(df_raw)

            # 3. Guardar resultados
            output_path = os.path.join(output_dir, f"{truck_id}_sensor.csv")
            df_clean.write_csv(output_path)

            # 4. Mostrar resumen ejecución
            print(f"✅ Dataset transformer ejecutado con exito para {truck_id}")
            print("📋 Primeras 2 filas:")
            print(df_clean.head(2))

        except Exception as e:
            print(f"❌ Error procesando {truck_id}: {str(e)}")


if __name__ == "__main__":
    # Configurar rutas de importación
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    run_sensor_etl_for_all_trucks()
    print("\n✅ Proceso completado para todos los camiones")
