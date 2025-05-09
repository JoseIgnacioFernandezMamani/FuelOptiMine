from transform.implementation.sensor.transformer import SensorTransformer
from extract.implementations.local.csv_extractor import CSVExtractor
from extract.config.settings import DATA_DIR
import sys
import os


def run_sensor_etl_pipeline():
    # Configuración de rutas y parámetros
    truck_id = "T-210"
    dataset_name = "train_data"
    data_type = "sensor"

    # Configurar rutas de importación
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    print(f"🚚 Iniciando ETL para camión {truck_id} - {dataset_name} (Sensor Data)")

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

        # Obtener DataFrame de Polars directamente
        df_raw = raw_data[data_type]
        print(f"✅ Datos crudos cargados: {len(df_raw)} registros")
        print("Esquema inicial:", df_raw.schema)

        # 2. Transformación de datos
        print("\n🔄 Procesando datos con SensorTransformer...")
        transformer = SensorTransformer()
        df_clean = transformer.run_transform(df_raw)

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("La transformación devolvió datos vacíos")

        # 3. Verificación de nuevas columnas

        """expected_columns = [
            'slope_percent', 'fuel_consumption', 'idle_time',
            'distance_traveled', 'efficiency_ratio', 'slope_impact'
        ]
        missing_columns = [col for col in expected_columns if col not in df_clean.columns]
        if missing_columns:
            raise ValueError(f"Columnas esperadas faltantes: {missing_columns}") 
        """

        # 4. Reporte de métricas actualizado
        print("\n📊 Métricas finales:")
        metrics = [
            ("Registros iniciales", "initial_records"),
            ("Registros limpios", "cleaned_records"),
            ("Registros nulos removidos", "removed_null_records"),
            ("Registros duplicados removidos", "removed_duplicate_records"),
            ("Registros inválidos de esquema", "invalid_schema_records"),
            ("Registros con combustible inválido", "invalid_fuel_records"),
            ("Porcentaje limpio", "clean_data_percentage"),
        ]

        for nombre, clave in metrics:
            valor = transformer.metrics.get(clave, "N/A")
            if isinstance(valor, float):
                print(f"- {nombre}: {valor:.2f}%")
            else:
                print(f"- {nombre}: {valor}")

        # 5. Muestra de resultados con nuevas columnas
        print("\n🔍 Muestra de datos transformados (Primeras 5 filas):")
        sample_columns = [
            "distance_traveled",
            "consumption_rate",
            "fuel_consumption",
            "refuel_event",
            "slope_percent",
            "slope_impact",
            "efficiency_ratio",
        ]
        print(df_clean.select(sample_columns).head(105))

        # 6. Análisis adicional de las nuevas características

        """ print("\n📈 Estadísticas clave de las nuevas columnas:")
        stats = df_clean.select([
            pl.col('fuel_consumption').mean().alias('consumo_promedio(l/h)'),
            pl.col('efficiency_ratio').mean().alias('eficiencia_promedio(m/l)'),
            pl.col('slope_impact').value_counts()
        ])
        print(stats) """

    except Exception as e:
        print(f"\n❌ Error crítico en el pipeline: {str(e)}")
        if hasattr(e, "__traceback__"):
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    run_sensor_etl_pipeline()
