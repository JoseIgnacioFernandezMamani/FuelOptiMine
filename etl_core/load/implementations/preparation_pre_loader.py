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
    df_fuel_supply: pl.DataFrame,
    df_cycle: pl.DataFrame,
) -> pl.DataFrame:
    """
    Unifica los dataframes de diferentes fuentes en base al timestamp de los sensores.
    Lógica de unión:
    - Para time_model y fuel_supply: última entrada <= timestamp del sensor
    - Para cycle: ciclo activo donde E_TravelingStart <= timestamp del sensor
    """

    # Función auxiliar para convertir a datetime si es necesario
    def ensure_datetime(df, col_name):
        if df[col_name].dtype == pl.String:
            return df.with_columns(pl.col(col_name).str.to_datetime())
        return df

    # Diagnóstico detallado de df_cycle
    print("🔍 Diagnóstico detallado de df_cycle:")
    print(f"  Forma: {df_cycle.shape}")
    print(f"  Columnas: {df_cycle.columns}")
    print(f"  Tipos de datos:")
    for col in df_cycle.columns:
        dtype = df_cycle[col].dtype
        nulls = df_cycle[col].null_count()
        print(f"    {col}: {dtype} (nulls: {nulls})")

    # Verificar valores nulos en columnas críticas
    null_traveling = df_cycle["E_TravelingStart"].null_count()
    null_unloading = df_cycle["L_UnloadingEnd"].null_count()

    print(f"  E_TravelingStart nulls: {null_traveling}")
    print(f"  L_UnloadingEnd nulls: {null_unloading}")

    if null_traveling > 0:
        print("⚠️ Registros con E_TravelingStart nulo:")
        null_records = df_cycle.filter(pl.col("E_TravelingStart").is_null())
        print(
            null_records.select(
                [
                    "Equipment",
                    "TruckFleet",
                    "ShiftDate",
                    "Shift",
                    "E_TravelingStart",
                    "L_UnloadingEnd",
                ]
            )
        )

        # Eliminar registros con E_TravelingStart nulo
        print("🧹 Eliminando registros con E_TravelingStart nulo...")
        df_cycle = df_cycle.filter(pl.col("E_TravelingStart").is_not_null())
        print(f"  Registros restantes: {len(df_cycle)}")

    if null_unloading > 0:
        print("⚠️ Registros con L_UnloadingEnd nulo:")
        null_records = df_cycle.filter(pl.col("L_UnloadingEnd").is_null())
        print(
            null_records.select(
                [
                    "Equipment",
                    "TruckFleet",
                    "ShiftDate",
                    "Shift",
                    "E_TravelingStart",
                    "L_UnloadingEnd",
                ]
            )
        )

        # Eliminar registros con L_UnloadingEnd nulo
        print("🧹 Eliminando registros con L_UnloadingEnd nulo...")
        df_cycle = df_cycle.filter(pl.col("L_UnloadingEnd").is_not_null())
        print(f"  Registros restantes: {len(df_cycle)}")

    # Convertir columnas de tiempo a datetime solo si son string
    df_sensor = ensure_datetime(df_sensor, "TimeStamp")
    df_time_model = ensure_datetime(df_time_model, "TimeStamp")
    df_fuel_supply = ensure_datetime(df_fuel_supply, "TimeStamp")

    # Para df_cycle, verificar ambas columnas
    if df_cycle["E_TravelingStart"].dtype == pl.String:
        df_cycle = df_cycle.with_columns(pl.col("E_TravelingStart").str.to_datetime())
    if df_cycle["L_UnloadingEnd"].dtype == pl.String:
        df_cycle = df_cycle.with_columns(pl.col("L_UnloadingEnd").str.to_datetime())

    # Verificar que no queden nulls después de la conversión
    final_null_traveling = df_cycle["E_TravelingStart"].null_count()
    final_null_unloading = df_cycle["L_UnloadingEnd"].null_count()

    print(f"🔍 Después de conversión datetime:")
    print(
        f"  E_TravelingStart tipo: {df_cycle['E_TravelingStart'].dtype}, nulls: {final_null_traveling}"
    )
    print(
        f"  L_UnloadingEnd tipo: {df_cycle['L_UnloadingEnd'].dtype}, nulls: {final_null_unloading}"
    )

    # Validar que tengamos datos para procesar
    if len(df_cycle) == 0:
        print("❌ No hay registros válidos en df_cycle después de limpiar nulls")
        return pl.DataFrame()

    print("🔗 Iniciando uniones...")

    # 1. Unión con time_model (último registro <= timestamp del sensor)
    print("  1️⃣ Uniendo con time_model...")
    df_unified = df_sensor.join_asof(
        df_time_model.sort("TimeStamp"),  # Asegurar ordenamiento
        on="TimeStamp",
        by=["Equipment", "TruckFleet", "ShiftDate", "Shift"],
        strategy="backward",
        suffix="_time_model",
    )
    print(f"     Registros después de unión time_model: {len(df_unified)}")

    # 2. Unión con fuel_supply (último registro <= timestamp del sensor)
    print("  2️⃣ Uniendo con fuel_supply...")
    df_unified = df_unified.join_asof(
        df_fuel_supply.sort("TimeStamp"),  # Asegurar ordenamiento
        on="TimeStamp",
        by=["Equipment", "TruckFleet", "ShiftDate", "Shift"],
        strategy="backward",
        suffix="_fuel_supply",
    )
    print(f"     Registros después de unión fuel_supply: {len(df_unified)}")

    # 3. Unión con cycle (ciclo activo en el momento del sensor)
    print("  3️⃣ Uniendo con cycle...")
    try:
        df_cycle_sorted = df_cycle.sort("E_TravelingStart")
        df_unified = df_unified.join_asof(
            df_cycle_sorted,
            left_on="TimeStamp",
            right_on="E_TravelingStart",
            by=["Equipment", "TruckFleet", "ShiftDate", "Shift"],
            strategy="backward",
        )
        print(f"     Registros después de unión cycle: {len(df_unified)}")

        # Filtrar solo ciclos activos
        print("  🔍 Filtrando ciclos activos...")
        df_unified = df_unified.filter(
            (pl.col("TimeStamp") >= pl.col("E_TravelingStart"))
            & (pl.col("TimeStamp") <= pl.col("L_UnloadingEnd"))
        )
        print(f"     Registros después de filtro ciclos activos: {len(df_unified)}")

    except Exception as e:
        print(f"❌ Error en unión con cycle: {str(e)}")
        # Mostrar muestra de datos para diagnóstico
        print("📊 Muestra de df_cycle:")
        print(
            df_cycle_sorted.head(3).select(
                [
                    "Equipment",
                    "TruckFleet",
                    "ShiftDate",
                    "Shift",
                    "E_TravelingStart",
                    "L_UnloadingEnd",
                ]
            )
        )
        raise

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
        transformer_sensor = SensorTransformer()
        transformer_cycle = CycleTransformer()
        transformer_fuel_supply = FuelSupplyTransformer()
        transformer_time_model = TimeModelTransformer()

        # Transformar datos
        df_sensor = transformer_sensor.run_transform(raw_data["sensor"])
        df_cycle = transformer_cycle.run_transform(raw_data["cycle"])
        df_fuel_supply = transformer_fuel_supply.run_transform(raw_data["fuel_supply"])
        df_time_model = transformer_time_model.run_transform(raw_data["time_model"])

        # Debug: Verificar tipos de datos
        print("🔍 Tipos de datos después de transformación:")
        print(f"  Sensor TimeStamp: {df_sensor['TimeStamp'].dtype}")
        print(f"  Time Model TimeStamp: {df_time_model['TimeStamp'].dtype}")
        print(f"  Fuel Supply TimeStamp: {df_fuel_supply['TimeStamp'].dtype}")
        print(f"  Cycle E_TravelingStart: {df_cycle['E_TravelingStart'].dtype}")
        print(f"  Cycle L_UnloadingEnd: {df_cycle['L_UnloadingEnd'].dtype}")

        # Unificar dataframes
        unified_df = unify_dataframes(
            df_sensor=df_sensor,
            df_time_model=df_time_model,
            df_fuel_supply=df_fuel_supply,
            df_cycle=df_cycle,
        )

        # Guardar en CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"unified_data_{truck_id}_{timestamp}.csv"
        output_path = os.path.join(os.getcwd(), output_filename)

        unified_df.write_csv(output_path)

        print(f"✅ Datos unificados guardados en: {output_filename}")
        print(f"📊 Total de registros: {len(unified_df)}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
