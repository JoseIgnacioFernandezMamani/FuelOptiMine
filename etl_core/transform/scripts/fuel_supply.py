from etl_core.transform.implementation.fuel_supply.transformer import (
    FuelSupplyTransformer,
)
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.utils.equipment_constants import TRUCK_SPECS
import sys
import os
import polars as pl


def run_fuel_supply_etl_for_all_trucks():
    # Crear carpeta para resultados
    output_dir = "frontend/web/app/output"
    os.makedirs(output_dir, exist_ok=True)

    # Filtrar camiones (excluyendo T-210 y T-211)
    trucks = ["T-210"]

    print(f"🚚 Procesando {len(trucks)} camiones para datos de abastecimiento...")

    for truck_id in trucks:
        print(f"\n{'='*50}")
        print(f"⛽  Iniciando ETL para {truck_id} (Fuel Supply)")
        print(f"{'='*50}")

        try:
            # 1. Extracción de datos
            extractor: CSVExtractor = CSVExtractor("train_data", truck_id)
            raw_data: dict[str, pl.DataFrame] = extractor.load_data()
            df_raw: pl.DataFrame = raw_data["fuel_supply"]

            # 2. Transformación
            transformer: FuelSupplyTransformer = FuelSupplyTransformer()
            df_clean: pl.DataFrame = transformer.run_transform(df_raw)

            # 3. Guardar resultados
            output_path = os.path.join(output_dir, f"{truck_id}_fuel_supply.csv")
            df_clean.write_csv(output_path)

            # 4. Mostrar resumen ejecución
            print(f"✅ Dataset transformer ejecutado con exito para {truck_id}")
            print("📋 Primeras 2 filas:")
            print(df_clean.columns)

        except Exception as e:
            print(f"❌ Error procesando {truck_id}: {str(e)}")


if __name__ == "__main__":
    # Configurar rutas de importación
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    run_fuel_supply_etl_for_all_trucks()
    print("\n✅ Proceso completado para todos los camiones")
