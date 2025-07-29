from etl_core.transform.implementation.fuel_supply.transformer import (
    FuelSupplyTransformer,
)
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
import sys
import os
import polars as pl


def run_fuel_supply_etl_test():
    # Configuración
    truck_id = "T-210"
    dataset_name = "train_data"
    data_type = "fuel_supply"

    # Agregar ruta al core del proyecto
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    print(f"🚚 Iniciando ETL para {truck_id} - {dataset_name} (Fuel Supply Data)")

    try:
        # 1. Extracción
        print("\n🔍 Extrayendo datos desde CSV...")
        extractor = CSVExtractor(dataset_name, truck_id)
        raw_data = extractor.load_data()

        if not raw_data or data_type not in raw_data:
            raise ValueError(
                f"No se encontró '{data_type}' o la estructura es inválida"
            )

        df_raw = raw_data[data_type]
        print(f"✅ Datos cargados: {df_raw.height} registros")
        print("Esquema inicial:", df_raw.schema)

        # 2. Transformación
        print("\n🔄 Procesando datos con FuelSupplyTransformer...")
        transformer = FuelSupplyTransformer()
        df_clean = transformer.run_transform(df_raw)

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("La transformación devolvió datos vacíos")

        # 3. Verificación de columnas mínimas esperadas
        expected_columns = [
            "Origin",
            "ShiftDate",
            "TimeStamp",
            "Equipment",
            "TruckFleet",
            "FuelLevelLiters",
            "Shift",
            "FuelLevel",
        ]
        missing_columns = [
            col for col in expected_columns if col not in df_clean.columns
        ]
        if missing_columns:
            raise ValueError(f"Faltan columnas esperadas: {missing_columns}")

        # 4. Métricas actualizadas
        print("\n📊 Métricas del transformador:")
        metrics = [
            ("Registros iniciales", "initial_records"),
            ("Registros luego de limpieza", "cleaned_records"),
            ("Registros nulos eliminados", "removed_null_records"),
            ("Registros duplicados eliminados", "removed_duplicate_records"),
            ("Modelos inválidos", "invalid_truck_models"),
            ("Outliers eliminados", "outliers_removed"),
            ("Orígenes inválidos", "invalid_origin_records"),
            ("Campos categóricos reemplazados", "categorical_null_empty_replaced"),
            ("Porcentaje de datos limpios", "clean_data_percentage"),
            ("Porcentaje de datos finales", "final_data_percentage"),
        ]

        for nombre, clave in metrics:
            valor = transformer.metrics.get(clave, "N/A")
            if isinstance(valor, float):
                print(f"- {nombre}: {valor:.2f}%")
            else:
                print(f"- {nombre}: {valor}")

        # 5. Muestra de los datos finales
        print("\n🔍 Muestra de datos transformados (5 filas):")
        print(df_clean.select(expected_columns).head(5))

        # 6. Guardar resultados (opcional)
        """
        output_path = os.path.join(os.getcwd(), f"{truck_id}_fuel_supply_transformed.csv")
        df_clean.write_csv(output_path)
        print(f"\n💾 Resultados guardados en: {output_path}")
        """

    except Exception as e:
        print(f"\n❌ Error crítico en el pipeline: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_fuel_supply_etl_test()
