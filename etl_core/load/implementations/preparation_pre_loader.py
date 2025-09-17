import polars as pl
import os
from datetime import datetime
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.transform import (
    CycleTransformer,
    FuelSupplyTransformer,
    SensorTransformer,
    TimeModelTransformer,
)


def unify_dataframes(
    df_sensor: pl.DataFrame,
    df_time_model: pl.DataFrame,
    df_cycle: pl.DataFrame,
    max_tolerance_days: int = 365,
) -> pl.DataFrame:
    if len(df_sensor) == 0 or len(df_time_model) == 0 or len(df_cycle) == 0:
        raise ValueError("One or more required dataframes are empty")

    # ordenar dataframes
    df_sensor = df_sensor.sort(["TruckFleet", "Equipment", "TimeStamp"])
    df_time_model = df_time_model.sort(["TruckFleet", "Equipment", "TimeStamp"])
    df_cycle = df_cycle.sort(["TruckFleet", "Equipment", "TimeStampIni"])

    # Unir sensor con time_model
    df_unified = df_sensor.join_asof(
        df_time_model,
        on="TimeStamp",
        strategy="backward",
        suffix="_tm",
        tolerance=f"{max_tolerance_days}d",
        coalesce=False,
        allow_parallel=True,
    )
    df_unified = df_unified.join_asof(
        df_cycle,
        left_on="TimeStamp",
        right_on="TimeStampIni",
        strategy="backward",
        suffix="_cycle",
        tolerance=f"{max_tolerance_days}d",
        coalesce=False,
        allow_parallel=True,
    )

    ## obtener los timemodelid que no aparecen en unified_df
    df_missing_tm = df_time_model.join(
        df_unified.select("TimeModelId").unique(),
        on="TimeModelId",
        how="anti",
    )
    df_missing_cycle = df_cycle.join(
        df_unified.select("CycleId", "StageSequence").unique(),
        on=["CycleId", "StageSequence"],
        how="anti",
    )

    # Obtener columnas originales
    time_model_columns = df_time_model.columns
    cycle_columns = df_cycle.columns

    # Crear diccionarios de renombrado dinámicamente SOLO para las columnas que aparecen duplicadas
    # Las columnas únicas (como CycleId, TimeModelId, StageSequence, etc.) NO se renombran

    # Encontrar columnas que ya tienen sufijo en df_unified (fueron renombradas automáticamente)
    cols_with_tm_suffix = [col for col in df_unified.columns if col.endswith("_tm")]
    cols_with_cycle_suffix = [
        col for col in df_unified.columns if col.endswith("_cycle")
    ]

    # Crear mapeo solo para las columnas que necesitan renombrado
    # (las que fueron renombradas automáticamente en el join)
    original_tm_cols = [col.replace("_tm", "") for col in cols_with_tm_suffix]
    original_cycle_cols = [col.replace("_cycle", "") for col in cols_with_cycle_suffix]

    rename_map_tm = {col: f"{col}_tm" for col in original_tm_cols}
    rename_map_cycle = {col: f"{col}_cycle" for col in original_cycle_cols}

    df_missing_tm_renamed = df_missing_tm.rename(rename_map_tm)
    df_missing_cycle_renamed = df_missing_cycle.rename(rename_map_cycle)

    # columnas que tiene el unified DESPUÉS de todos los joins
    unified_cols = df_unified.columns

    # asegurar que df_missing_tm_renamed tenga todas las columnas de unified
    missing_cols_tm = [
        col for col in unified_cols if col not in df_missing_tm_renamed.columns
    ]
    if missing_cols_tm:
        df_missing_tm_renamed = df_missing_tm_renamed.with_columns(
            [pl.lit(None).alias(col) for col in missing_cols_tm]
        )

    # asegurar que df_missing_cycle_renamed tenga todas las columnas de unified
    missing_cols_cycle = [
        col for col in unified_cols if col not in df_missing_cycle_renamed.columns
    ]
    if missing_cols_cycle:
        df_missing_cycle_renamed = df_missing_cycle_renamed.with_columns(
            [pl.lit(None).alias(col) for col in missing_cols_cycle]
        )

    # reordenar columnas para que el orden coincida exactamente
    df_missing_tm_renamed = df_missing_tm_renamed.select(unified_cols)
    df_missing_cycle_renamed = df_missing_cycle_renamed.select(unified_cols)

    # Concatenar todos los dataframes
    df_unified = pl.concat(
        [df_unified, df_missing_tm_renamed, df_missing_cycle_renamed], how="vertical"
    )

    # Crear columna temporal para ordenamiento y ordenar
    df_unified = df_unified.with_columns(
        pl.coalesce(
            [pl.col("TimeStamp"), pl.col("TimeStamp_tm"), pl.col("TimeStampIni")]
        ).alias("SortTimestamp")
    ).sort("SortTimestamp")

    # obtener todas las columnas
    exclude_columns = [
        "TimeModelId",
        "CycleId",
        "SortTimestamp",
        "Equipment",
        "TruckFleet",
        "RecordDuration",
        "Shift",
        "ShiftDate",
        "FuelGauge",
        "Speed",
        "RPM",
        "Ralenti",
        "Latitude",
        "Longitude",
        "Elevation",
        "SpeedAvg",
        "Acceleration",
        "DistanceTraveled",
        "SlopePercent",
        "Status",
        "Category",
        "Event",
        "Shovel",
        "ShovelModel",
    ]
    columns = df_unified.columns
    columns_filter = [col for col in columns if col not in exclude_columns]

    # Eliminar columnas duplicadas genericas
    df_unified = df_unified.with_columns(
        pl.when(pl.col(col) == pl.col(col).shift(-1))
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
        for col in columns_filter
    )

    # Eliminar columnas fecha de forma especial
    df_unified = df_unified.with_columns(
        pl.when(pl.col("StageSequence").is_null())
        .then(None)
        .otherwise(pl.col("TimeStampIni"))
        .alias("TimeStampIni"),
    )

    df_unified = df_unified.with_columns(
        pl.col("ShiftDate").forward_fill(),
        pl.col("Shift").forward_fill(),
        pl.col("TimeStamp").forward_fill(),
        pl.col("RecordDuration").fill_null(0),
        pl.col("Equipment").forward_fill(),
        pl.col("TruckFleet").forward_fill(),
        pl.when(pl.col("FuelLevelLiters") >= 0)
        .then(pl.col("FuelLevelLiters"))
        .otherwise(None)
        .forward_fill()
        .alias("FuelLevelLiters"),
        pl.when(pl.col("FuelLevel") >= 0)
        .then(pl.col("FuelLevel"))
        .otherwise(None)
        .forward_fill()
        .alias("FuelLevel"),
        pl.col("FuelGauge").forward_fill(),
        pl.col("Speed").forward_fill(),
        pl.col("RPM").forward_fill(),
        pl.col("Ralenti").forward_fill(),
        pl.col("Latitude").forward_fill(),
        pl.col("Longitude").forward_fill(),
        pl.col("Elevation").forward_fill(),
        pl.col("TimeModelId").forward_fill(),
        pl.col("CycleId").forward_fill(),
        pl.col("Status").forward_fill(),
        pl.col("Category").forward_fill(),
        pl.col("Event").forward_fill(),
        pl.col("SpeedAvg").fill_null(0),
        pl.col("Acceleration").fill_null(0),
        pl.col("DistanceTraveled").fill_null(0),
        pl.col("SlopePercent").fill_null(0),
        pl.when(pl.col("StageSequence") == 4)
        .then(pl.col("Shovel"))
        .otherwise(None)
        .alias("Shovel"),
        pl.when(pl.col("StageSequence") == 4)
        .then(pl.col("ShovelModel"))
        .otherwise(None)
        .alias("ShovelModel"),
    )

    df_unified = df_unified.drop(
        [
            "Equipment_tm",
            "Equipment_cycle",
            "TruckFleet_tm",
            "TruckFleet_cycle",
            "Shift_cycle",
            "ShiftDate_cycle",
            "ShiftDate_tm",
            "Shift_tm",
        ]
    )

    return df_unified


def main():
    """Función principal que ejecuta la unificación y guarda el resultado en CSV"""

    # Configuración
    truck_id = "T-210"
    dataset_name = "train_data"

    try:
        # Extraer datos
        extractor = CSVExtractor(dataset_name, truck_id)
        raw_data = extractor.load_data()

        # Instanciar transformadores
        transformer_sensor = SensorTransformer(truck_id=truck_id)
        transformer_cycle = CycleTransformer()

        transformer_time_model = TimeModelTransformer()

        # Transformar datos
        df_sensor = transformer_sensor.run_transform(raw_data["sensor"])
        df_cycle = transformer_cycle.run_transform(raw_data["cycle"])
        df_time_model = transformer_time_model.run_transform(raw_data["time_model"])

        # Validar que los dataframes no sean None
        if df_sensor is None or df_time_model is None or df_cycle is None:
            print(
                "❌ Uno o más dataframes transformados son None. Verifique las fuentes de datos."
            )
            return

        # Unificar dataframes
        unified_df = unify_dataframes(
            df_sensor=df_sensor,
            df_time_model=df_time_model,
            df_cycle=df_cycle,
            max_tolerance_days=365,
        )

        if len(unified_df) == 0:
            print(
                "❌ No se pudieron unificar datos. Verificar compatibilidad de fuentes."
            )
            return

        # Guardar en CSV
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"unified_data_{truck_id}.csv"
        output_path = os.path.join(os.getcwd(), output_filename)

        unified_df.write_csv(output_path)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
