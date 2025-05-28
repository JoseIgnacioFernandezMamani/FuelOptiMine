import argparse
import polars as pl
from etl.extract.csv_extractor import CsvExtractor
from etl.transform.implementation.sensor.transformer import SensorTransformer
from etl.load.sensor_loader import SensorLoader
import logging
import time


def run_sensor_etl(file_path: str):
    """Ejecuta el pipeline completo para datos de sensores"""
    # 1. Extracción
    logging.info("Iniciando extracción...")
    extractor = CsvExtractor()
    df = extractor.read(file_path)

    # 2. Transformación
    logging.info("Transformando datos...")
    transformer = SensorTransformer()
    transformed_df = transformer.transform(df)

    # 3. Conversión a diccionarios
    records = transformed_df.to_dicts()

    # 4. Carga
    logging.info("Cargando a ClickHouse...")
    loader = SensorLoader(
        host="localhost",
        port=9000,
        user="default",
        password="password",
        database="default",
    )

    start_time = time.time()
    record_count = loader.load_data(records)
    elapsed = time.time() - start_time

    logging.info(f"✅ Carga completada: {record_count} registros")
    logging.info(f"⏱  Tiempo total: {elapsed:.2f} segundos")
    logging.info(f"🚀 Velocidad: {record_count/elapsed:.2f} registros/segundo")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="ETL Core Pipeline")
    parser.add_argument(
        "--type", required=True, choices=["sensor", "cycle", "fuel", "time"]
    )
    parser.add_argument("--file", required=True, help="Ruta al archivo de entrada")

    args = parser.parse_args()

    if args.type == "sensor":
        run_sensor_etl(args.file)
    elif args.type == "cycle":
        # run_cycle_etl(args.file)
        pass
    # ... otros tipos
