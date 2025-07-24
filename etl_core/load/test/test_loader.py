from etl_core.transform.implementation.sensor.transformer import SensorTransformer
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.load.implementations import ClickHouseLoader
import sys
import os
import logging
import polars as pl


def test_sensor_etl_pipeline():
    """Prueba de pipeline completo ETL para datos de sensor"""
    # Configuración de rutas y parámetros
    truck_id = "T-210"
    dataset_name = "train_data"
    data_type = "sensor"

    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("SensorETLTest")

    logger.info(
        f"🚚 Iniciando ETL para camión {truck_id} - {dataset_name} (Sensor Data)"
    )

    try:
        # 1. Extracción de datos
        logger.info("\n🔍 Extrayendo datos desde CSV...")
        extractor = CSVExtractor(dataset_name, truck_id)
        raw_data, metadata = extractor.load_data()

        if not raw_data or data_type not in raw_data:
            logger.error(f"Datos de tipo '{data_type}' no encontrados")
            return

        df_raw = raw_data[data_type]
        logger.info(f"✅ Datos crudos cargados: {len(df_raw)} registros")
        logger.info(f"Esquema inicial: {df_raw.schema}")

        # 2. Transformación de datos
        logger.info("\n🔄 Procesando datos con SensorTransformer...")
        transformer = SensorTransformer()
        df_clean = transformer.run_transform(df_raw)

        if df_clean is None or df_clean.is_empty():
            logger.error("Transformación devolvió datos vacíos")
            return

        logger.info(f"✅ Datos transformados: {len(df_clean)} registros")
        logger.info(f"Esquema transformado: {df_clean.schema}")

        # 3. Preparar datos para ClickHouse
        logger.info("\n🔧 Preparando datos para carga...")
        # Convertir columnas booleanas a UInt8
        bool_cols = [
            col for col, dtype in df_clean.schema.items() if dtype == pl.Boolean
        ]
        if bool_cols:
            df_clean = df_clean.with_columns(
                [pl.col(col).cast(pl.UInt8) for col in bool_cols]
            )
            logger.info(f"Columnas booleanas convertidas: {bool_cols}")

        # 4. Carga de datos
        df_clean = df_clean.select(
            [
                pl.col("ShiftDate").cast(pl.Date),
                pl.col("Shift").cast(pl.Utf8),
                pl.col("TimeStamp").dt.convert_time_zone("UTC"),
                pl.col("RecordDuration").cast(pl.Float64),
                pl.col("Equipment").cast(pl.Utf8),
                pl.col("TruckFleet").cast(pl.Utf8),
                pl.col("FuelLevel").cast(pl.Float64),
                pl.col("FuelLevelLiters").cast(pl.Float64),
                pl.col("FuelGauge").cast(pl.Utf8),
                pl.col("Speed").cast(pl.Float64),
                pl.col("RPM").cast(pl.Float64),
                pl.col("Ralenti").cast(pl.Utf8),
                pl.col("Latitude").cast(pl.Float64),
                pl.col("Longitude").cast(pl.Float64),
                pl.col("Elevation").cast(pl.Float64),
            ]
        )

        logger.info("\n📤 Cargando datos transformados a ClickHouse...")
        with ClickHouseLoader() as loader:
            success = loader.load_dataset(data_type, df_clean)

            if success:
                logger.info("✅ Carga completada exitosamente!")
                metrics = loader.get_metrics()
                logger.info(f"Total filas cargadas: {metrics['total_rows']}")
                logger.info(f"Duración: {metrics['last_duration']:.2f} segundos")
                logger.info(f"Filas/segundo: {metrics['avg_rps']:.2f}")
            else:
                logger.error("❌ Fallo en la carga de datos")

    except Exception as e:
        logger.exception(f"❌ Error en el pipeline: {str(e)}")


if __name__ == "__main__":
    test_sensor_etl_pipeline()
