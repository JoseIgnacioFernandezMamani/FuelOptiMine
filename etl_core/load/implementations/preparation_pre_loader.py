import polars as pl
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Importar dependencias
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.transform import (
    CycleTransformer,
    SensorTransformer,
    TimeModelTransformer,
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataUnifier:
    """
    High-level data integration component that consolidates multiple operational sources—
    sensor telemetry, time model events, and cycle metrics—into a single, analysis-ready dataset.

    This class prepares unified data before loading into the database.

    The unified output is used as a foundational dataset for downstream analytics and database storage.
    """

    def __init__(self, max_tolerance_days: int = 365):
        """
        Initializes the DataUnifier with configurable temporal join tolerance.

        Args:
            max_tolerance_days (int, optional): The maximum allowed time difference (in days)
                when performing temporal "as-of" joins between data sources.
                This tolerance ensures proper alignment of asynchronous sensor,
                time model, and cycle records without losing relevant matches.
                Defaults to 365.
        """
        self.max_tolerance_days = max_tolerance_days
        self.tolerance = f"{max_tolerance_days}d"

        # define exactly the columns we need from each DataFrame
        self.sensor_columns = [
            "ShiftDate",
            "Shift",
            "TimeStamp",
            "Equipment",
            "TruckFleet",
            "FuelLevelLiters",
            "Latitude",
            "Longitude",
            "Elevation",
            "SpeedAvg",
            "Acceleration",
            "SlopePercent",
            "ValidFuel",
            "DeltaFuel",
            "BeforeAvg",
            "AfterAvg",
        ]

        self.time_model_columns = [
            "TimeModelId",
            "TimeStamp",
            "Status",
            "Category",
            "Event",
        ]

        self.cycle_columns = [
            "CycleId",
            "Shovel",
            "ShovelModel",
            "StageType",
            "StageSequence",
            "TimeStampIni",
            "TimeStampFin",
            "LoadingZone",
            "Material",
            "MeasuredTonnage",
            "ReportedTonnage",
            "DestinationType",
            "Destination",
            "Distance",
            "Latitude",
            "Longitude",
            "Elevation",
            "TimeEfficiencyPercentage",
        ]

        # Configure core columns that are not processed for duplicates
        self.core_columns = {
            # sensor
            "ShiftDate",
            "Shift",
            "TimeStamp",
            "Equipment",
            "TruckFleet",
            "FuelLevelLiters",
            "Latitude",
            "Longitude",
            "Elevation",
            "SpeedAvg",
            "Acceleration",
            "SlopePercent",
            "ValidFuel",
            "DeltaFuel",
            "BeforeAvg",
            "AfterAvg",
            # time model
            "TimeStamp_tm",
            "Status",
            "Category",
            "Event",
            # cycle
            "CycleId",
            "Shovel",
            "ShovelModel",
            "TimeStampIni",
            "TimeStampFin",
            # add SortTimestamp
            "SortTimestamp",
        }

        # Columnas para forward fill
        self.forward_fill_cols = {
            "ShiftDate",
            "Shift",
            "TimeStamp",
            "Equipment",
            "TruckFleet",
            "FuelLevelLiters",
            "Latitude",
            "Longitude",
            "Elevation",
            "SpeedAvg",
            "Acceleration",
            "SlopePercent",
            "CycleId",
        }

    def _validate_dataframes(
        self,
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ) -> None:
        """
        Validates input DataFrames to ensure they contain valid data for processing.

        Args:
            df_sensor (pl.DataFrame)
            df_time_model (pl.DataFrame)
            df_cycle (pl.DataFrame)

        Raises:
            ValueError: If all DataFrames are empty or are contain very few records.
        """
        if df_sensor is None or len(df_sensor) <= 10000:
            raise ValueError(
                "DataFrame de sensores no cumple con la minima cantidad de datos"
            )

        if df_time_model is None or len(df_time_model) <= 30:
            raise ValueError(
                "DataFrame de time_model no cumple con la minima cantidad de datos"
            )

        if df_cycle is None or len(df_cycle) <= 30:
            raise ValueError(
                "DataFrame de cycle no cumple con la minima cantidad de datos"
            )

        logger.info(
            f"Validación OK - Sensor: {len(df_sensor)}, TimeModel: {len(df_time_model)}, Cycle: {len(df_cycle)} filas"
        )

    def _perform_joins(
        self,
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Performs temporal asof joins on sensor, time model, and cycle DataFrames.

        Args:
            df_sensor (pl.DataFrame): Sensor telemetry data, sorted by timestamp.
            df_time_model (pl.DataFrame): Time model data, sorted by timestamp.
            df_cycle (pl.DataFrame): Cycle operational data, sorted by end timestamp.

        Returns:
            pl.DataFrame: Resulting DataFrame after sequentially joining df_sensor with df_time_model and df_cycle using 'forward' asof joins.
        """

        # Join with time_model
        result = df_sensor.join_asof(
            df_time_model,
            on="TimeStamp",
            strategy="forward",
            suffix="_tm",
            tolerance=self.tolerance,
            coalesce=False,
            allow_parallel=True,
        )

        # Join with cycle
        result = result.join_asof(
            df_cycle,
            left_on="TimeStamp",
            right_on="TimeStampFin",
            strategy="forward",
            suffix="_cycle",
            tolerance=self.tolerance,
            coalesce=False,
            allow_parallel=True,
        )
        return result

    def _get_missing_records(
        self,
        df_unified: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Identifies missing records in the unified DataFrame compared to time model and cycle DataFrames.

        Args:
            df_unified (pl.DataFrame): Unified DataFrame containing combined records.
            df_time_model (pl.DataFrame): Time model DataFrame with TimeModelId column.
            df_cycle (pl.DataFrame): Cycle DataFrame with CycleId and StageSequence columns.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Two DataFrames containing missing records from
            the time model and cycle DataFrames respectively.
        """
        df_missing_tm = pl.DataFrame()
        df_missing_cycle = pl.DataFrame()

        # missing records from time_model
        unified_tm_ids = df_unified.select("TimeModelId").unique()
        df_missing_tm = df_time_model.join(unified_tm_ids, on="TimeModelId", how="anti")

        # missing records from cycle
        unified_cycle_ids = df_unified.select("CycleId", "StageSequence").unique()
        df_missing_cycle = df_cycle.join(
            unified_cycle_ids, on=["CycleId", "StageSequence"], how="anti"
        )

        # get unified columns
        unified_cols = df_unified.columns

        # get columns with suffix _tm from the unified dataframe
        tm_suffix_cols = [col for col in unified_cols if col.endswith("_tm")]

        # get columns with suffix _cycle from the unified dataframe
        cycle_suffix_cols = [col for col in unified_cols if col.endswith("_cycle")]

        # mapping for time_model
        rename_map_tm = {
            col.replace("_tm", ""): col
            for col in tm_suffix_cols
            if col.replace("_tm", "") in df_missing_tm.columns
        }
        df_missing_tm = (
            df_missing_tm.rename(rename_map_tm) if rename_map_tm else df_missing_tm
        )

        # mapping for cycle
        rename_map_cycle = {
            col.replace("_cycle", ""): col
            for col in cycle_suffix_cols
            if col.replace("_cycle", "") in df_missing_cycle.columns
        }

        df_missing_cycle = (
            df_missing_cycle.rename(rename_map_cycle)
            if rename_map_cycle
            else df_missing_cycle
        )

        # Add missing columns with None
        for df, name in [(df_missing_tm, "tm"), (df_missing_cycle, "cycle")]:
            missing_cols = [col for col in unified_cols if col not in df.columns]
            if missing_cols:
                df = df.with_columns([pl.lit(None).alias(col) for col in missing_cols])
            df = df.select(unified_cols)

            if name == "tm":
                df_missing_tm = df
            else:
                df_missing_cycle = df

        return df_missing_tm, df_missing_cycle

    def _remove_consecutive_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Skips columns defined as core columns in the class to process in _apply_business_logic.

        Compares each column’s value with the next row’s value and replaces consecutive duplicates with None,
        keeping only the last occurrence in any sequence of repeated values.

        Args:
            df (pl.DataFrame): Input unified DataFrame to process.

        Returns:
            pl.DataFrame: DataFrame with consecutive duplicates removed.
        """
        if len(df) == 0:
            return df

        columns_to_process = [col for col in df.columns if col not in self.core_columns]
        if not columns_to_process:
            return df

        return df.with_columns(
            [
                pl.when((pl.col(col) == pl.col(col).shift(-1)))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
                for col in columns_to_process
            ]
        )

    def _apply_fill_strategies(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies batch strategies to fill missing values in specified columns of the DataFrame.

        - Performs forward fill on columns listed in FORWARD_FILL_COLS that exist in the DataFrame.

        Args:
            df (pl.DataFrame): Input unified DataFrame.

        Returns:
            pl.DataFrame: DataFrame with fill values.
        """
        if len(df) == 0:
            return df

        # Forward fill
        forward_cols = [col for col in self.forward_fill_cols if col in df.columns]
        if forward_cols:
            df = df.with_columns([pl.col(col).forward_fill() for col in forward_cols])

        return df

    def _apply_business_logic(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies domain-specific business logic to enrich or filter columns based on stage sequences.

        Updates columns like Shovel, ShovelModel, TimeStampIni, TimeStampFin, and CycleId only for records
        with relevant stage sequences (1 to 8). Columns not existing in the DataFrame are added with None values
        to keep schema consistency.

        Args:
            df (pl.DataFrame): Input DataFrame to which business rules are applied.

        Returns:
            pl.DataFrame: Transformed DataFrame with business logic conditions applied.
        """
        if len(df) == 0 or "StageSequence" not in df.columns:
            return df

        business_expressions = []

        # Logic for Shovel and ShovelModel (only in StageSequence 4)
        for col_name in ["Shovel", "ShovelModel"]:
            if col_name in df.columns:
                business_expressions.append(
                    pl.when(pl.col("StageSequence") == 4)
                    .then(pl.col(col_name))
                    .otherwise(None)
                    .alias(col_name)
                )
            else:
                business_expressions.append(pl.lit(None).alias(col_name))

        # Lógica para timestamps y cycle (StageSequence 1-8)
        valid_stages = [1, 2, 3, 4, 5, 6, 7, 8]
        for col_name in ["TimeStampIni", "TimeStampFin", "CycleId"]:
            if col_name in df.columns:
                business_expressions.append(
                    pl.when(pl.col("StageSequence").is_in(valid_stages))
                    .then(pl.col(col_name))
                    .otherwise(None)
                    .alias(col_name)
                )
            else:
                business_expressions.append(pl.lit(None).alias(col_name))

        # logic for time model columns (if TimeModelId is present)
        for col_name in ["TimeStamp_tm", "Status", "Category", "Event"]:
            if col_name in df.columns:
                business_expressions.append(
                    pl.when(pl.col("TimeModelId").is_not_null())
                    .then(pl.col(col_name))
                    .otherwise(None)
                    .alias(col_name)
                )
            else:
                business_expressions.append(pl.lit(None).alias(col_name))

        return df.with_columns(business_expressions) if business_expressions else df

    def unify(
        self,
        df_sensor: pl.DataFrame,
        df_time_model: pl.DataFrame,
        df_cycle: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Unifies multiple operational data sources into a single, coherent, and analysis-ready DataFrame.

        This function serves as the core of the ETL pipeline, merging sensor telemetry,
        time model patterns, and operational cycle data into a chronologically aligned,
        cleaned, and enriched dataset. It performs rigorous validation, temporal alignment,
        deduplication, and business-specific transformations to ensure that the resulting
        DataFrame preserves both temporal integrity and operational context.

        The unification process consists of the following steps:
            1. **Validation** – Ensures that all input DataFrames contain sufficient and valid data.
            2. **Column selection & sorting** – Filters only the required columns and sorts each DataFrame chronologically.
            3. **Temporal joins (as-of joins)** – Combines records based on temporal proximity using a configurable tolerance window.
            4. **Missing record detection** – Identifies records present in the original sources but missing after the join process.
            5. **Duplicate cleanup** – Removes consecutive duplicate values to reduce noise without losing relevant information.
            6. **Reintegration of missing data** – Concatenates previously missing records back into the unified dataset.
            7. **Global sorting** – Creates a unified timestamp (`SortTimestamp`) to temporally order all records.
            8. **Filling and enrichment** – Applies forward-fill strategies and business logic transformations based on operational stages.
            9. **Final output** – Returns a comprehensive, enriched, temporally aligned dataset ready for downstream analytics or modeling.

        Args:
            df_sensor (pl.DataFrame): Sensor telemetry data including position, fuel, and performance metrics.
            df_time_model (pl.DataFrame): Temporal state and event data from the time model.
            df_cycle (pl.DataFrame): Operational cycle data including load/haul metrics and stage sequences.

        Returns:
            pl.DataFrame: A fully unified, chronologically ordered, business-enriched DataFrame ready for advanced analysis.
        """
        logger.info("Iniciando unificación de dataframes")

        # 1. validate
        self._validate_dataframes(df_sensor, df_time_model, df_cycle)

        # 2. Seleccionar solo las columnas necesarias y ordenar dataframes
        sensor_cols = [col for col in self.sensor_columns if col in df_sensor.columns]
        tm_cols = [
            col for col in self.time_model_columns if col in df_time_model.columns
        ]
        cycle_cols = [col for col in self.cycle_columns if col in df_cycle.columns]

        df_sensor = df_sensor.select(sensor_cols).sort("TimeStamp")
        df_time_model = df_time_model.select(tm_cols).sort("TimeStamp")
        df_cycle = (
            df_cycle.filter(pl.col("StageSecuence") != 1)
            .select(cycle_cols)
            .sort("TimeStampFin")
        )
        df_cycle_st1 = df_cycle.filter(pl.col("StageSecuence") == 1)
        # 3. temporal joins
        df_unified = self._perform_joins(df_sensor, df_time_model, df_cycle)

        if len(df_unified) == 0:
            logger.error("No se generaron datos después de los joins temporales")
            raise ValueError("No se generaron datos después de los joins temporales")

        # 4. handling missing records
        df_missing_tm, df_missing_cycle = self._get_missing_records(
            df_unified, df_time_model, df_cycle
        )

        # 5. Process main data
        df_unified = self._remove_consecutive_duplicates(df_unified)

        # 6. Concatenate missing records
        df_unified = pl.concat(
            [df_unified, df_missing_tm, df_missing_cycle], how="vertical"
        )

        # 7. sort final dataframe
        df_unified = df_unified.with_columns(
            pl.min_horizontal(["TimeStamp", "TimeStamp_tm", "TimeStampFin"]).alias(
                "SortTimestamp"
            )
        ).sort("SortTimestamp")

        # 8. apply fill strategies and business logic
        df_unified = self._apply_fill_strategies(df_unified)
        df_unified = self._apply_business_logic(df_unified)

        logger.info(f"Unificación completada. Filas resultantes: {len(df_unified)}")
        return df_unified


def main():
    """Función principal simplificada."""
    truck_id = "T-210"
    dataset_name = "train_data"

    try:
        logger.info(f"Iniciando procesamiento para {truck_id}")

        # Extraer datos
        extractor = CSVExtractor(dataset_name, truck_id)
        raw_data = extractor.load_data()

        # Transformar datos
        transformers = {
            "sensor": SensorTransformer(truck_id=truck_id),
            "cycle": CycleTransformer(),
            "time_model": TimeModelTransformer(),
        }

        transformed_data = {}
        for data_type, transformer in transformers.items():
            result = transformer.run_transform(raw_data[data_type])
            if result is None or len(result) == 0:
                raise ValueError(f"Transformación de {data_type} falló o está vacía")

            transformed_data[data_type] = result
            logger.info(f"Transformación {data_type}: {len(result)} filas")

        # Unify data
        unifier = DataUnifier(max_tolerance_days=365)
        unified_df = unifier.unify(
            df_sensor=transformed_data["sensor"],
            df_time_model=transformed_data["time_model"],
            df_cycle=transformed_data["cycle"],
        )

        # Guardar resultado
        output_filename = f"unified_data_{truck_id}.csv"
        unified_df.write_csv(output_filename)

        logger.info(f"Procesamiento completado. Archivo: {output_filename}")
        logger.info(
            f"Estadísticas: {len(unified_df)} filas, {len(unified_df.columns)} columnas"
        )

        # Mostrar columnas
        logger.info("Columnas disponibles:")
        for i, col in enumerate(unified_df.columns[:10]):
            logger.info(f"   • {col}")
        if len(unified_df.columns) > 10:
            logger.info(f"   • ... y {len(unified_df.columns) - 10} columnas más")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
