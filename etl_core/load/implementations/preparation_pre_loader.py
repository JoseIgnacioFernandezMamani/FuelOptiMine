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
    - Para time_model: última entrada <= timestamp del sensor
    - Para fuel_supply: último registro por equipo (sin restricción temporal)
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

    # Diagnóstico de rangos temporales
    print("🔍 Diagnóstico de rangos temporales:")
    sensor_min_time = df_sensor["TimeStamp"].min()
    sensor_max_time = df_sensor["TimeStamp"].max()
    fuel_min_time = df_fuel_supply["TimeStamp"].min()
    fuel_max_time = df_fuel_supply["TimeStamp"].max()

    print(f"  Rango temporal sensor: {sensor_min_time} a {sensor_max_time}")
    print(f"  Rango temporal fuel_supply: {fuel_min_time} a {fuel_max_time}")

    # Verificar si hay gap temporal
    if fuel_max_time < sensor_min_time:
        gap_days = (sensor_min_time - fuel_max_time).days
        print(
            f"  ⚠️ GAP TEMPORAL: {gap_days} días entre último fuel_supply y primer sensor"
        )
        print("  📝 Aplicando estrategia de último registro por equipo")

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

    # 2. Unión con fuel_supply (último registro por equipo - SIN restricción temporal)
    print("  2️⃣ Uniendo con fuel_supply (último registro por equipo)...")

    # Obtener el último registro de fuel_supply por equipo
    latest_fuel_supply = (
        df_fuel_supply.sort("TimeStamp", descending=True)
        .group_by(["Equipment", "TruckFleet"])
        .agg(
            pl.all().first()
        )  # Toma el primer registro (que es el más reciente tras el sort)
    )

    print(f"     Registros únicos de fuel_supply por equipo: {len(latest_fuel_supply)}")

    df_unified = df_unified.join(
        latest_fuel_supply,
        on=["Equipment", "TruckFleet"],
        how="left",
        suffix="_fuel_supply",
    )

    # Verificar éxito de la unión
    fuel_supply_data_count = df_unified["FuelLevelLiters_fuel_supply"].null_count()
    total_records = len(df_unified)
    success_rate = ((total_records - fuel_supply_data_count) / total_records) * 100

    print(f"     Registros después de unión fuel_supply: {len(df_unified)}")
    print(
        f"     Registros con datos de fuel_supply: {total_records - fuel_supply_data_count}/{total_records} ({success_rate:.1f}%)"
    )

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

    # Verificación final
    print("🔍 Verificación final de datos:")
    fuel_final_nulls = df_unified["FuelLevelLiters_fuel_supply"].null_count()
    fuel_final_success = ((len(df_unified) - fuel_final_nulls) / len(df_unified)) * 100
    print(f"  Datos de fuel_supply completos: {fuel_final_success:.1f}%")

    if fuel_final_nulls > 0:
        print(f"  ⚠️ Aún hay {fuel_final_nulls} registros sin datos de fuel_supply")
    else:
        print("  ✅ Todos los registros tienen datos de fuel_supply")

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

        # Debug: Verificar cantidades de registros
        print("🔍 Cantidades de registros por fuente:")
        print(f"  Sensor: {len(df_sensor)} registros")
        print(f"  Time Model: {len(df_time_model)} registros")
        print(f"  Fuel Supply: {len(df_fuel_supply)} registros")
        print(f"  Cycle: {len(df_cycle)} registros")

        # Unificar dataframes
        unified_df = unify_dataframes(
            df_sensor=df_sensor,
            df_time_model=df_time_model,
            df_fuel_supply=df_fuel_supply,
            df_cycle=df_cycle,
        )

        if len(unified_df) == 0:
            print(
                "❌ No se pudieron unificar datos. Verificar compatibilidad de fuentes."
            )
            return

        # Mostrar muestra de datos unificados
        print("🔍 Muestra de datos unificados (primeras 3 filas):")
        sample_columns = [
            "TimeStamp",
            "Equipment",
            "FuelLevel",
            "Speed",
            "FuelLevelLiters_fuel_supply",
            "Status",
            "Shovel",
        ]
        available_columns = [col for col in sample_columns if col in unified_df.columns]
        print(unified_df.select(available_columns).head(3))

        # Guardar en CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"unified_data_{truck_id}_{timestamp}.csv"
        output_path = os.path.join(os.getcwd(), output_filename)

        unified_df.write_csv(output_path)

        print(f"✅ Datos unificados guardados en: {output_filename}")
        print(f"📊 Total de registros: {len(unified_df)}")
        print(f"📊 Total de columnas: {len(unified_df.columns)}")

        # Estadísticas finales
        print("\n📈 Estadísticas de completitud:")
        fuel_nulls = unified_df["FuelLevelLiters_fuel_supply"].null_count()
        fuel_completeness = ((len(unified_df) - fuel_nulls) / len(unified_df)) * 100
        print(f"  Datos de fuel_supply: {fuel_completeness:.1f}% completos")

        if "Status" in unified_df.columns:
            status_nulls = unified_df["Status"].null_count()
            status_completeness = (
                (len(unified_df) - status_nulls) / len(unified_df)
            ) * 100
            print(f"  Datos de time_model: {status_completeness:.1f}% completos")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
