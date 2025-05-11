from pathlib import Path
import polars as pl
from etl_core.etl.transform.implementation.time_model.transformer import (
    TimeModelTransformer,
)
from etl_core.etl.extract.implementations.local.csv_extractor import CSVExtractor
import sys
import os


def run_time_model_etl_pipeline():
    # Configuración de rutas y parámetros
    truck_id = "T-210"
    dataset_name = "train_data"
    data_type = "time_model"

    # Configurar rutas de importación
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    print(f"⏱️ Iniciando ETL para modelo temporal - {dataset_name}")

    try:
        # 1. Extracción de datos
        print("\n🔍 Extrayendo datos desde CSV...")
        extractor = CSVExtractor(dataset_name, truck_id)

        raw_data, metadata = extractor.load_data()

        if not raw_data or data_type not in raw_data:
            raise ValueError(f"Datos de tipo '{data_type}' no encontrados")

        df_raw = raw_data[data_type]
        print(f"✅ Datos crudos cargados: {len(df_raw)} registros")
        print("Esquema inicial:", df_raw.schema)

        # 2. Transformación base (limpieza + normalización)
        print("\n🔄 Procesando datos con TimeModelTransformer (solo limpieza)...")
        transformer = TimeModelTransformer()
        df_clean = transformer.run_transform(df_raw)

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("La limpieza/normalización falló")

        # 3. Validación de campos obligatorios dinámicos
        mandatory_columns = transformer.mandatory_columns  # Obtenido del esquema
        missing_columns = [
            col for col in mandatory_columns if col not in df_clean.columns
        ]
        if missing_columns:
            raise ValueError(f"Columnas obligatorias faltantes: {missing_columns}")

        # 4. Métricas de limpieza base
        print("\n📊 Métricas de limpieza base:")
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

        # 5. Validación de tipos de datos
        print("\n🔍 Validación de esquema final:")
        schema_checks = {
            "ShiftDate": pl.Date,
            "TimeStamp": pl.Datetime,
            "Equipment": pl.Utf8,
            "RecordDuration": pl.Float64,
        }

        for col, dtype in schema_checks.items():
            if df_clean.schema[col] != dtype:
                raise TypeError(
                    f"Tipo incorrecto en {col}: Esperado {dtype}, Obtenido {df_clean.schema[col]}"
                )

        # 6. Muestra de datos normalizados
        print("\n📄 Muestra de datos normalizados:")
        print(
            df_clean.select(
                ["ShiftDate", "TimeStamp", "Equipment", "Status", "RecordDuration"]
            ).head(3)
        )

    except Exception as e:
        print(f"\n❌ Error crítico: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_time_model_etl_pipeline()
