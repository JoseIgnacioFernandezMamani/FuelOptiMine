import polars as pl
import logging
from pathlib import Path
from typing import Optional
import json


class CycleDataProcessor:
    """
    Procesador de datos de ciclos de camiones mineros.

    Esta clase se encarga de cargar datos de sensores de camiones desde un archivo CSV,
    transformarlos para identificar ciclos operacionales y calcular métricas de consumo
    de combustible, guardando los resultados procesados en un archivo CSV de salida.
    """

    def __init__(
        self,
        input_file: str = "unified_data_T-210.csv",
        output_file: str = "cycles_data_processed.csv",
    ):
        """
        Inicializa el procesador de datos de ciclos.

        Args:
            input_file (str): Ruta al archivo CSV de entrada con datos de sensores
            output_file (str): Ruta al archivo CSV de salida para datos procesados
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.cycles_data = None

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Carga datos desde archivo CSV con manejo de valores nulos y errores.

        Procesa las columnas de fecha y timestamp, ordenando los datos por timestamp
        para el procesamiento secuencial de ciclos.
        """
        self.logger.info(f"Cargando datos desde {self.input_file}...")

        try:
            df = pl.read_csv(
                self.input_file,
                try_parse_dates=True,
            )

            self.df = df.sort("SortTimestamp")
            self.logger.info(f"Datos cargados exitosamente: {len(self.df)} registros")

        except Exception as e:
            self.logger.error(f"Error cargando datos: {str(e)}")
            raise

    def transform_cycles_data(self):
        """
        Procesa datos para identificar ciclos y calcular métricas de consumo de combustible.

        Este método realiza una transformación integral de datos para identificar ciclos
        operacionales de camiones y calcular el consumo de combustible por ciclo.
        Procesa datos de sensores para crear agregaciones significativas basadas en ciclos
        para entrenamiento de modelos.

        La transformación incluye:
            - Identificación de ciclos basada en secuencias de etapas
            - Suavizado de nivel de combustible usando mediana móvil
            - Unificación de datos geográficos
            - Agrupación y agregación de ciclos
            - Cálculo de consumo de combustible
            - Filtrado de calidad de datos

        Los resultados se almacenan en self.cycles_data como DataFrame de Polars.
                Procesa datos para identificar bloques basados en transiciones de stages.

        Este método identifica 3 tipos de bloques:
        1. Stage 1->4: Desde StageSequence=1 hasta StageSequence=4 (ciclo de carga/descarga)
        2. Stage 4->8: Desde StageSequence=4 hasta StageSequence=8 (ciclo de retorno/posicionamiento)
        3. Stage 8->1: Desde StageSequence=8 hasta StageSequence=1 (tiempo libre/sin ciclo)

        Los timestamps se extraen de los registros donde están presentes los stages específicos.
        """
        if self.df is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        self.logger.info("Iniciando transformación de datos de ciclos...")

        df = self.df.with_columns(
            [
                # rolling median del nivel de combustible
                pl.col("FuelLevelLiters")
                .cast(pl.Float64)
                .rolling_median(window_size=10, min_samples=3, center=True)
                .fill_null(0)
                .alias("MedianFuelLevelLiters"),
                # unificar destinos
                pl.coalesce([pl.col("LoadingZone"), pl.col("Destination")]).alias(
                    "Destination"
                ),
                # datos geográficos
                pl.when(pl.col("Latitude_cycle") != 0)
                .then(pl.col("Latitude_cycle"))
                .otherwise(pl.col("Latitude"))
                .alias("Latitude"),
                pl.when(pl.col("Longitude_cycle") != 0)
                .then(pl.col("Longitude_cycle"))
                .otherwise(pl.col("Longitude"))
                .alias("Longitude"),
                pl.when(pl.col("Elevation_cycle") != 0)
                .then(pl.col("Elevation_cycle"))
                .otherwise(pl.col("Elevation"))
                .alias("Elevation"),
                # para obtener la mediana aproximada
                pl.col("FuelLevelLiters").diff().abs().alias("delta_fuel"),
            ]
        )

        df = df.with_columns(
            # identificar ciclos
            pl.when(
                (pl.col("StageSequence") == 4)
                | (pl.col("StageSequence") == 8)
                | (pl.col("StageSequence") == 1)
            )
            .then(True)
            .otherwise(False)
            .alias("cycle_end")
        )

        # crear grupos de ciclos, se obtuvo los registros enteros de cada grupo
        df = df.with_columns(
            [
                pl.col("cycle_end")
                .shift(1, fill_value=False)
                .cum_sum()
                .alias("cycle_group")
            ]
        )

        result = (
            df.group_by("cycle_group")
            .agg(
                [
                    # metadatos
                    pl.len().alias(
                        "RecordsInCycle"
                    ),  # numero de registros de sensor, metadato
                    pl.col("TimeStampIni").last().alias("TimeStampIni"),
                    pl.col("TimeStampFin").last().alias("TimeStampFin"),
                    pl.col("ShiftDate").last().alias("ShiftDate"),
                    pl.col("Shift").last().alias("Shift"),
                    pl.col("Equipment").last().alias("Equipment"),
                    pl.col("TruckFleet")
                    .last()
                    .alias("TruckFleet"),  # fin de los metadatos
                    # modelo de estados
                    pl.col("Status").eq("OPERATIVO").sum().alias("OperativeCount"),
                    pl.col("Status").eq("DEMORA").sum().alias("DelayCount"),
                    pl.col("Status")
                    .eq("MANTENIMIENTO")
                    .sum()
                    .alias("MaintenanceCount"),
                    pl.col("Status").eq("RESERVA").sum().alias("ReserveCount"),
                    # modelo de categorias
                    # OPERATIVO
                    pl.col("Category").eq("EFECTIVO").sum().alias("EffectiveCount"),
                    # DEMORA
                    pl.col("Category")
                    .eq("D_PROGRAMADA")
                    .sum()
                    .alias("ScheduledDelayCount"),
                    pl.col("Category")
                    .eq("D_NO_PROGRAMADA")
                    .sum()
                    .alias("UnscheduledDelayCount"),
                    # RESERVA
                    pl.col("Category")
                    .eq("RESERVA")
                    .sum()
                    .alias("CategoryReserveCount"),
                    # MANTENIMIENTO
                    pl.col("Category")
                    .eq("M_PROGRAMADO")
                    .sum()
                    .alias("ScheduledMaintenanceCount"),
                    pl.col("Category")
                    .eq("M_NO_PROGRAMADO")
                    .sum()
                    .alias("UnscheduledMaintenanceCount"),
                    # REPARACION
                    pl.col("Category").eq("REPARACION").sum().alias("RepairCount"),
                    # variables para calculo, consumo de combustible
                    pl.col("MedianFuelLevelLiters")
                    .last()
                    .alias("MedianFuelLevelLiters"),
                    # variables numericas para modelo
                    pl.col("SpeedAvg").mean().alias("AvgSpeed"),
                    pl.col("SlopePercent").mean().alias("AvgSlopePercent"),
                    pl.col("Acceleration").mean().alias("AvgAcceleration"),
                    pl.col("TimeEfficiencyPercentage")
                    .sum()
                    .alias("TimeEfficiencyPercentage"),
                    pl.col("Latitude").last().alias("Latitude"),
                    pl.col("Longitude").last().alias("Longitude"),
                    pl.col("Elevation").last().alias("Elevation"),
                    # variables categoricas
                    pl.col("StageSequence").last().alias("StageSequence"),
                    pl.col("Destination").last().alias("Destination"),
                    pl.col("DestinationType").last().alias("DestinationType"),
                    pl.col("Material").last().alias("Material"),
                    pl.col("Shovel").last().alias("Shovel"),
                    # datos reales de los ciclos
                    pl.col("MeasuredTonnage").sum().alias("TotalMeasuredTonnage"),
                    pl.col("Distance").sum().alias("Distance"),
                ]
            )
            .sort("cycle_group")
        )

        # obtener los verdaderos timestamps de inicio y fin del ciclo
        result = result.with_columns(
            # Dynamic timestamp selection initial
            pl.when(
                (pl.col("StageSequence") == 4) & (pl.col("StageSequence").shift(1) == 1)
            )
            .then(pl.col("TimeStampIni").shift(1))
            .when(
                (pl.col("StageSequence") == 8) & (pl.col("StageSequence").shift(1) == 4)
            )
            .then(pl.col("TimeStampFin").shift(1))
            .when(
                (pl.col("StageSequence") == 1) & (pl.col("StageSequence").shift(1) == 8)
            )
            .then(pl.col("TimeStampFin").shift(1))
            .otherwise(pl.col("TimeStampIni"))
            .alias("TimeStampIni"),
            # dynamic timestamp selection final
            pl.when(
                (pl.col("StageSequence") == 1) & (pl.col("StageSequence").shift(1) == 8)
            )
            .then(pl.col("TimeStampIni"))
            .otherwise(pl.col("TimeStampFin"))
            .alias("TimeStampFin"),
            # Combustible consumido (sin cambios)
            pl.col("MedianFuelLevelLiters").diff().abs().alias("FuelConsumed"),
            # ajustar TimeEfficiencyPercentage
            pl.when(
                (pl.col("StageSequence") == 4)
                & (pl.col("StageSequence").shift(1) == 1)
                & (pl.col("TimeEfficiencyPercentage").shift(1) > 0)
            )
            .then(
                pl.col("TimeEfficiencyPercentage").shift(1)
                + pl.col("TimeEfficiencyPercentage")
            )
            .otherwise(pl.col("TimeEfficiencyPercentage"))
            .alias("TimeEfficiencyPercentage"),
        )

        result = result.with_columns(
            pl.when(
                (
                    (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                    .dt.total_seconds()
                    .abs()
                    > 3600
                )
                | (
                    (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                    .dt.total_seconds()
                    .abs()
                    < 50
                )
                & (pl.col("AvgSpeed") > 0.1)
                & (pl.col("Distance") > 50)
                & (pl.col("Distance") <= 3600)
            )
            .then(
                # Duración = distancia(m) / velocidad(m/s)
                pl.min_horizontal(
                    [
                        (pl.col("Distance") / (pl.col("AvgSpeed") / 3.6)),
                        pl.lit(float(900)),
                    ]
                ).clip(lower_bound=180)
            )
            .otherwise(
                (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                .dt.total_seconds()
                .abs()
            )
            .alias("CycleDurationSeconds"),
            pl.when(pl.col("StageSequence") == 1)
            .then(0)
            .otherwise(pl.col("TimeEfficiencyPercentage"))
            .alias("TimeEfficiencyPercentage"),
            # Destino con limpieza mejorada
            pl.when(pl.col("Destination").str.strip_chars().str.len_bytes() > 2)
            .then(pl.col("Destination"))
            .otherwise(pl.lit("UNKNOWN"))
            .alias("Destination"),
            # Tipo de destino con valor por defecto
            pl.col("DestinationType")
            .fill_null(pl.lit("LoadingZone"))
            .alias("DestinationType"),
            # Material con valor por defecto
            pl.col("Material").fill_null(pl.lit("Empty")).alias("Material"),
            pl.col("Shovel").fill_null(pl.lit("Unknown")).alias("Shovel"),
        )

        result = result.filter(
            (
                (pl.col("StageSequence").is_in([4, 8]))
                & (pl.col("StageSequence").is_not_null())
            )
        )

        self.cycles_data = self.best_data_for_train(result).filter(
            (
                (pl.col("StageSequence") == 4)
                & (pl.col("Distance") > 0)
                & (pl.col("Distance").is_not_null())
                & (pl.col("Distance") < 5000)
                & (pl.col("CycleDurationSeconds") > 0)
                & (pl.col("CycleDurationSeconds").is_not_null())
                & (pl.col("CycleDurationSeconds") < 3600)
                & (pl.col("FuelConsumed") > 0)
                & (pl.col("FuelConsumed").is_not_null())
                & (pl.col("FuelConsumed") < 110)
            )
            | (
                (pl.col("StageSequence") == 8)
                & (pl.col("TotalMeasuredTonnage") > 0)
                & (pl.col("TotalMeasuredTonnage").is_not_null())
                & (pl.col("TotalMeasuredTonnage") < 300)
                & (pl.col("Distance") > 0)
                & (pl.col("Distance").is_not_null())
                & (pl.col("Distance") < 5000)
                & (pl.col("CycleDurationSeconds") > 0)
                & (pl.col("CycleDurationSeconds").is_not_null())
                & (pl.col("CycleDurationSeconds") < 3600)
                & (pl.col("FuelConsumed") > 0)
                & (pl.col("FuelConsumed").is_not_null())
                & (pl.col("FuelConsumed") < 110)
            )
        )
        self.logger.info(
            f"Transformación completada: {len(self.cycles_data)} ciclos procesados"
        )

    def best_data_for_train(self, result: pl.DataFrame) -> pl.DataFrame:

        # obtener dataframes separados por stage
        df_stage4 = result.filter(pl.col("StageSequence") == 4)
        df_stage8 = result.filter(pl.col("StageSequence") == 8)

        stats = {"stage4": {}, "stage8": {}}
        # columnas específicas por stage
        cols_stage4 = ["Distance", "CycleDurationSeconds", "FuelConsumed"]
        cols_stage8 = [
            "TotalMeasuredTonnage",
            "Distance",
            "CycleDurationSeconds",
            "FuelConsumed",
        ]

        # --- Stage 4 ---
        for col in cols_stage4:
            values = df_stage4.select(
                [
                    pl.col(col).quantile(0.10).alias("q1"),
                    pl.col(col).quantile(0.90).alias("q3"),
                ]
            ).row(0)

            q1, q3 = values

            stats["stage4"][col] = {
                "q1": q1,
                "q3": q3,
            }

        # --- Stage 8 ---
        for col in cols_stage8:
            values = df_stage8.select(
                [
                    pl.col(col).quantile(0.1).alias("q1"),
                    pl.col(col).quantile(0.90).alias("q3"),
                ]
            ).row(0)

            q1, q3 = values

            stats["stage8"][col] = {
                "q1": q1,
                "q3": q3,
            }

        # ordenar por fuelConsumed
        df_stage4 = df_stage4.sort("FuelConsumed")
        df_stage4 = df_stage4.with_columns(
            pl.col("Distance").diff().alias("delta_distance"),
            pl.col("CycleDurationSeconds").diff().alias("delta_duration"),
            pl.col("TotalMeasuredTonnage").diff().alias("delta_tonnage"),
        )
        df_stage8 = df_stage8.sort("FuelConsumed")
        df_stage8 = df_stage8.with_columns(
            pl.col("Distance").diff().alias("delta_distance"),
            pl.col("CycleDurationSeconds").diff().alias("delta_duration"),
            pl.col("TotalMeasuredTonnage").diff().alias("delta_tonnage"),
        )

        # aplicar filtro de buen dato o mal dato
        df_stage4 = df_stage4.with_columns(
            pl.when(
                (pl.col("FuelConsumed") <= stats["stage4"]["FuelConsumed"]["q1"])
                | (pl.col("FuelConsumed") >= stats["stage4"]["FuelConsumed"]["q3"])
                | (pl.col("Distance") <= stats["stage4"]["Distance"]["q1"])
                | (pl.col("Distance") >= stats["stage4"]["Distance"]["q3"])
                | (
                    pl.col("CycleDurationSeconds")
                    <= stats["stage4"]["CycleDurationSeconds"]["q1"]
                )
                | (
                    pl.col("CycleDurationSeconds")
                    >= stats["stage4"]["CycleDurationSeconds"]["q3"]
                )
                | (
                    # Regla de patrón de crecimiento: al menos uno debe crecer
                    ~((pl.col("delta_distance") > 0) | (pl.col("delta_duration") > 0))
                )
            )
            .then(False)
            .otherwise(True)
            .alias("QualityData")
        )

        df_stage8 = df_stage8.with_columns(
            pl.when(
                (pl.col("FuelConsumed") <= stats["stage8"]["FuelConsumed"]["q1"])
                | (pl.col("FuelConsumed") >= stats["stage8"]["FuelConsumed"]["q3"])
                | (pl.col("Distance") <= stats["stage8"]["Distance"]["q1"])
                | (pl.col("Distance") >= stats["stage8"]["Distance"]["q3"])
                | (
                    pl.col("CycleDurationSeconds")
                    <= stats["stage8"]["CycleDurationSeconds"]["q1"]
                )
                | (
                    pl.col("CycleDurationSeconds")
                    >= stats["stage8"]["CycleDurationSeconds"]["q3"]
                )
                | (
                    pl.col("TotalMeasuredTonnage")
                    <= stats["stage8"]["TotalMeasuredTonnage"]["q1"]
                )
                | (
                    pl.col("TotalMeasuredTonnage")
                    >= stats["stage8"]["TotalMeasuredTonnage"]["q3"]
                )
                | (
                    # Regla de patrón de crecimiento: al menos uno debe crecer
                    ~(
                        (pl.col("delta_distance") > 0)
                        | (pl.col("delta_duration") > 0)
                        | (pl.col("delta_tonnage") > 0)
                    )
                )
            )
            .then(False)
            .otherwise(True)
            .alias("QualityData")
        )

        concat_df = (
            pl.concat([df_stage4, df_stage8])
            .sort("cycle_group")
            .drop(["delta_distance", "delta_duration", "delta_tonnage"])
        )

        return concat_df

    def save_to_csv(self, output_file: Optional[str] = None):
        """
        Guarda los datos de ciclos procesados en un archivo CSV.

        Args:
            output_file (Optional[str]): Ruta del archivo de salida.
                                       Si es None, usa self.output_file
        """
        if self.cycles_data is None:
            raise ValueError(
                "No hay datos procesados para guardar. Ejecute transform_cycles_data() primero."
            )

        output_path = output_file or self.output_file

        try:
            self.cycles_data.write_csv(output_path)
            self.logger.info(f"Datos guardados exitosamente en: {output_path}")
            self.logger.info(f"Registros guardados: {len(self.cycles_data)}")
        except Exception as e:
            self.logger.error(f"Error guardando datos: {str(e)}")
            raise

    def process_all(self, output_file: Optional[str] = None) -> pl.DataFrame:
        """
        Ejecuta todo el pipeline de procesamiento de datos.

        Carga datos, los transforma y guarda el resultado en CSV.

        Args:
            output_file (Optional[str]): Ruta del archivo de salida

        Returns:
            pl.DataFrame: DataFrame con los ciclos procesados
        """
        self.logger.info("Iniciando procesamiento completo de datos de ciclos...")

        try:
            # Ejecutar pipeline completo
            self.load_data()
            self.transform_cycles_data()
            self.save_to_csv(output_file)

            self.logger.info("Procesamiento completo finalizado exitosamente")
            return self.cycles_data

        except Exception as e:
            self.logger.error(f"Error durante el procesamiento: {str(e)}")
            raise

    def get_summary_stats(self) -> dict:
        """
        Retorna estadísticas resumen de los datos procesados.

        Returns:
            dict: Diccionario con estadísticas clave de los ciclos
        """
        if self.cycles_data is None:
            raise ValueError(
                "No hay datos procesados. Ejecute transform_cycles_data() primero."
            )

        stats = {
            "total_cycles": len(self.cycles_data),
            "avg_fuel_consumed": self.cycles_data.select(
                pl.col("FuelConsumed").mean()
            ).item(),
            "avg_cycle_duration": self.cycles_data.select(
                pl.col("CycleDurationSeconds").mean()
            ).item(),
            "avg_distance": self.cycles_data.select(pl.col("Distance").mean()).item(),
            "unique_equipment": self.cycles_data.select(
                pl.col("Equipment").n_unique()
            ).item(),
            "date_range": {
                "start": self.cycles_data.select(pl.col("TimeStampIni").min()).item(),
                "end": self.cycles_data.select(pl.col("TimeStampFin").max()).item(),
            },
        }

        return stats


# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del procesador
    processor = CycleDataProcessor(
        input_file="unified_data_T-210.csv", output_file="cycles_data.csv"
    )

    try:
        # Procesar todos los datos
        cycles_df = processor.process_all()

        # Mostrar estadísticas
        stats = processor.get_summary_stats()
        print("\n=== ESTADÍSTICAS RESUMEN ===")
        print(f"Total de ciclos: {stats['total_cycles']}")
        print(
            f"Consumo promedio de combustible: {stats['avg_fuel_consumed']:.2f} litros"
        )
        print(f"Duración promedio de ciclo: {stats['avg_cycle_duration']:.2f} segundos")
        print(f"Distancia promedio: {stats['avg_distance']:.2f} metros")
        print(f"Equipos únicos: {stats['unique_equipment']}")
        print(
            f"Rango de fechas: {stats['date_range']['start']} a {stats['date_range']['end']}"
        )

        # Mostrar muestra de datos
        print(f"\n=== MUESTRA DE DATOS PROCESADOS ===")
        print(cycles_df.head())

    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
