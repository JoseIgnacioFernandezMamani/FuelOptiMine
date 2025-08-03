from etl_core.etl.transform.implementation.cycle.transformer import CycleTransformer
from etl_core.etl.extract.implementations.local.csv_extractor import CSVExtractor
import sys
import os


def run_cycle_etl_pipeline():
    # Configuración
    truck_id = "T-210"
    dataset_name = "train_data"
    data_type = "cycle"

    # Configurar rutas
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    print(f"🚚 Iniciando ETL para camión {truck_id} - {dataset_name} (Cycle Data)")

    try:
        # 1. Extracción (sin cambios)
        print("\n🔍 Extrayendo datos desde CSV...")
        extractor = CSVExtractor(dataset_name, truck_id)
        raw_data, metadata = extractor.load_data()

        if not raw_data or data_type not in raw_data:
            raise ValueError(f"Datos de tipo '{data_type}' no encontrados")

        df_raw = raw_data[data_type]
        print(f"✅ Datos crudos cargados: {len(df_raw)} registros")
        print("Esquema inicial:", df_raw.schema)

        # 2. Transformación Base (solo limpieza + normalización)
        print("\n🔄 Procesando datos con CycleTransformer (solo limpieza)...")
        transformer = CycleTransformer()
        df_clean = transformer.run_transform(
            df_raw
        )  # <-- Solo ejecuta hasta normalize_and_validate

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("La limpieza/normalización falló")

        # 3. Validación de columnas (usando propiedad dinámica)
        mandatory_columns = (
            transformer.mandatory_columns
        )  # <-- Dinámico desde el esquema
        missing_columns = [
            col for col in mandatory_columns if col not in df_clean.columns
        ]
        if missing_columns:
            raise ValueError(f"Columnas obligatorias faltantes: {missing_columns}")

        # 4. Métricas Base (actualizadas)
        print("\n📊 Métricas de limpieza/normalización:")
        base_metrics = [
            ("Registros iniciales", "initial_records"),
            ("Registros limpios", "cleaned_records"),
            ("Registros nulos eliminados", "removed_null_records"),
            ("Registros duplicados eliminados", "removed_duplicate_records"),
            ("Porcentaje válido", "clean_data_percentage"),
        ]

        for nombre, clave in base_metrics:
            valor = transformer.metrics.get(clave, "N/A")
            if isinstance(valor, float):
                print(f"- {nombre}: {valor:.2f}%")
            else:
                print(f"- {nombre}: {valor}")

        # 5. Muestra de datos normalizados (sin transformaciones hijas)
        print("\n🔍 Muestra de datos normalizados (Primeras 5 filas):")
        # Columnas del esquema base (no incluye campos calculados)
        sample_columns = [
            "ShiftDate",
            "Equipment",
            "G_Latitude",
            "G_Longitude",
            "D_Latitude",
            "D_Longitude",
        ]
        print(df_clean.select(sample_columns).head(5))

    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_cycle_etl_pipeline()
