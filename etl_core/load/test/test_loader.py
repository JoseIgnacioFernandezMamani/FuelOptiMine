from etl_core.transform.implementation.sensor.transformer import SensorTransformer
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.transform import (
    CycleTransformer,
    SensorTransformer,
    TimeModelTransformer,
)
from etl_core.load.implementations import ClickHouseLoader
import sys
import os
import logging
from logging import Logger
import polars as pl
import gc


def setup_logging():
    """Configurar logging simple"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("ETL_Pipeline")


def process_single_truck(truck_id: str, dataset_name: str = "train_data") -> bool:
    """Procesar un solo camión con limpieza de memoria"""
    logger = setup_logging()

    try:
        logger.info(f"Iniciando {truck_id}")

        # EXTRACCIÓN
        logger.info(f"{truck_id} - Extrayendo datos")
        extractor = CSVExtractor(dataset_name, truck_id)
        raw_data = extractor.load_data()

        # Validar datos
        required_types = ["sensor", "cycle", "time_model"]
        for data_type in required_types:
            if data_type not in raw_data or raw_data[data_type].is_empty():
                logger.error(f"{truck_id} - Datos '{data_type}' faltantes")
                return False

        # TRANSFORMACIÓN
        logger.info(f"{truck_id} - Transformando datos")

        transformers = {
            "sensor": SensorTransformer(truck_id=truck_id),
            "cycle": CycleTransformer(),
            "time_model": TimeModelTransformer(),
        }

        transformed_data = {}

        for data_type, transformer in transformers.items():
            result = transformer.run_transform(raw_data[data_type])

            if result is None or result.is_empty():
                logger.error(f"{truck_id} - Transformación {data_type} falló")
                return False

            transformed_data[data_type] = result

            # Limpiar datos raw después de transformar
            del raw_data[data_type]
            gc.collect()

        # Limpiar raw_data completamente
        del raw_data
        gc.collect()

        # CARGA
        logger.info(f"{truck_id} - Cargando a ClickHouse")

        with ClickHouseLoader(max_tolerance_days=365) as loader:
            success = loader.load_unified_data(
                df_sensor=transformed_data["sensor"],
                df_time_model=transformed_data["time_model"],
                df_cycle=transformed_data["cycle"],
            )

            if success:
                metrics = loader.get_metrics()
                logger.info(
                    f"{truck_id} - Completado: {metrics.get('total_rows', 0):,} filas"
                )
                return True
            else:
                logger.error(f"{truck_id} - Fallo en carga")
                return False

    except Exception as e:
        logger.error(f"{truck_id} - Error: {str(e)}")
        return False

    finally:
        # Limpieza agresiva de memoria
        if "transformed_data" in locals():
            del transformed_data
        if "extractor" in locals():
            del extractor
        if "transformers" in locals():
            del transformers
        gc.collect()


def run_all_trucks():
    """Ejecutar ETL para todos los camiones secuencialmente"""
    # "T-210",
    truck_ids = [
        "T-210",
        "T-233",
        "T-232",
        "T-231",
        "T-230",
        "T-225",
        "T-224",
        "T-223",
        "T-222",
        "T-221",
        "T-220",
        "T-219",
        "T-218",
        "T-217",
        "T-216",
        "T-215",
        "T-214",
        "T-213",
        "T-212",
        "T-211",
        "T-236",
        "T-237",
        "T-238",
        "T-240",
        "T-241",
        "T-242",
        "T-243",
    ]

    logger = setup_logging()
    logger.info(f"Procesando {len(truck_ids)} camiones secuencialmente")

    successful = 0
    failed = 0
    failed_trucks = []

    for i, truck_id in enumerate(truck_ids, 1):
        logger.info(f"[{i}/{len(truck_ids)}] Procesando {truck_id}")

        success = process_single_truck(truck_id)

        if success:
            successful += 1
            logger.info(f"{truck_id} - EXITOSO")
        else:
            failed += 1
            failed_trucks.append(truck_id)
            logger.error(f"{truck_id} - FALLIDO")

        # Limpieza entre camiones
        gc.collect()

        # Progreso cada 5 camiones
        if i % 5 == 0:
            logger.info(
                f"Progreso: {i}/{len(truck_ids)} - Exitosos: {successful}, Fallidos: {failed}"
            )

    # Resumen final
    logger.info("=" * 50)
    logger.info("RESUMEN FINAL")
    logger.info(f"Total: {len(truck_ids)}")
    logger.info(f"Exitosos: {successful}")
    logger.info(f"Fallidos: {failed}")
    logger.info(f"Tasa éxito: {successful/len(truck_ids)*100:.1f}%")

    if failed_trucks:
        logger.error(f"Camiones fallidos: {', '.join(failed_trucks)}")

    return successful >= len(truck_ids) * 0.8  # 80% éxito mínimo


if __name__ == "__main__":
    success = run_all_trucks()
    sys.exit(0 if success else 1)
