from etl_core.transform.implementation.fuel_supply.transformer import (
    FuelSupplyTransformer,
)
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
import sys
import os


def run_fuel_supply_etl_pipeline():
    # Configuración de rutas y parámetros
    truck_id = "T-211"
    dataset_name = "train_data"
    data_type = "fuel_supply"

    columns_to_keep = [
        "Origin",
        "ShiftDate",
        "TimeStamp",
        "Equipment",
        "TruckFleet",
        "FuelLevelLiters",
        "LastRefuel",
        "Shift",
        "FuelLevel",
    ]

    # Configurar rutas de importación
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    print(f"📦 Iniciando ETL para despachos - {dataset_name}")

    try:
        # 1. Extracción de datos
        print("\n🔍 Extrayendo datos desde CSV...")
        extractor = CSVExtractor(dataset_name, truck_id)

        # Cargar datos y manejar posibles errores
        raw_data, metadata = extractor.load_data()

        if not raw_data or data_type not in raw_data:
            raise ValueError(
                f"Datos de tipo '{data_type}' no encontrados o estructura inválida"
            )

        # Obtener DataFrame de Polars
        df_raw = raw_data[data_type]
        print(f"✅ Datos crudos cargados: {len(df_raw)} registros")
        print("Esquema inicial:", df_raw.schema)

        # 2. Transformación de datos
        # 2. Transformación base (solo limpieza + normalización)
        print("\n🔄 Procesando datos con BaseTransformer...")  #
        transformer = FuelSupplyTransformer()  # Usa tu clase hija
        df_clean = transformer.run_transform(
            df_raw
        )  # Solo ejecuta hasta normalize_and_validate

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("La limpieza/normalización devolvió datos vacíos")

        # 4. Reporte de métricas (ajustado a las disponibles) ⚡
        print("\n📊 Métricas de limpieza/normalización:")
        base_metrics = [
            ("Registros iniciales", "initial_records"),
            ("Registros limpios", "cleaned_records"),
            ("Registros nulos eliminados", "removed_null_records"),
            ("Registros duplicados eliminados", "removed_duplicate_records"),
            ("Porcentaje limpio", "clean_data_percentage"),
        ]

        for nombre, clave in base_metrics:
            valor = transformer.metrics.get(clave, "N/A")
            if isinstance(valor, float):
                print(f"- {nombre}: {valor:.2f}%")
            else:
                print(f"- {nombre}: {valor}")

        # 5. Muestra de datos procesados (solo esquema normalizado)
        print("\n🔍 Muestra de datos normalizados:")
        print(df_clean.head(5))
        # print("Esquema final:", df_clean.schema)

        df_selected = df_clean.select(columns_to_keep)

        # Ruta de guardado
        output_path = (
            "/mnt/d/Develop/FuelOptiMine/frontend/web/app/output/T-211_fuel_supply.csv"
        )

        # Guardar como CSV
        df_selected.write_csv(output_path)

        print(f"\n💾 Archivo guardado en: {output_path}")

    except Exception as e:
        print(f"\n❌ Error crítico en el pipeline: {str(e)}")
        if hasattr(e, "__traceback__"):
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    run_fuel_supply_etl_pipeline()
