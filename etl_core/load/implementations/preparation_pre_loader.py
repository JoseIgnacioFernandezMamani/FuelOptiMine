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
    max_tolerance_days: int = 365,
) -> pl.DataFrame:

    if (
        len(df_sensor) == 0
        or len(df_time_model) == 0
        or len(df_fuel_supply) == 0
        or len(df_cycle) == 0
    ):
        raise ValueError("One or more required dataframes are empty")

    # ordenar dataframes
    df_sensor = df_sensor.sort(["TruckFleet", "Equipment", "TimeStamp"])
    df_time_model = df_time_model.sort(["TruckFleet", "Equipment", "TimeStamp"])
    df_fuel_supply = df_fuel_supply.sort(["TruckFleet", "Equipment", "TimeStamp"])
    # df_cycle = df_cycle.sort(["TruckFleet", "Equipment", "TimeStamp"])

    # Unir sensor con time_model
    df_unified = df_sensor.join_asof(
        df_time_model,
        on="TimeStamp",
        strategy="forward",
        suffix="_tm",
        tolerance=f"{max_tolerance_days}d",
        coalesce=False,
        allow_parallel=True,
    )

    """ 
    df_unified = df_unified.join_asof(
        df_fuel_supply,
        on="TimeStamp",
        strategy="forward",
        suffix="_fs",
        tolerance=f"{max_tolerance_days}d",
        coalesce=False,
        allow_parallel=True,
    )
    """

    """
    df_unified = df_unified.join_asof(
        df_cycle.sort("E_TravelingStart"),
        left_on="TimeStamp",
        right_on="E_TravelingStart",
        by=["TruckFleet", "Equipment"],
        strategy="forward",
    )
    """
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

        # Unificar dataframes
        unified_df = unify_dataframes(
            df_sensor=df_sensor,
            df_time_model=df_time_model,
            df_fuel_supply=df_fuel_supply,
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
