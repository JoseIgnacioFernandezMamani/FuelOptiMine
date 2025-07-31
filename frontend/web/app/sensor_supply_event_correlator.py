import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import date, timedelta
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize_scalar
from scipy.optimize import OptimizeResult
from datetime import datetime, timedelta


class SensorSupplyEventCorrelator:
    """
    Correlaciona eventos de refill (sensores) con eventos de fuel supply
    bajo criterios específicos de agrupación temporal y emparejamiento.
    """

    def __init__(self, truck_id: str) -> None:
        self.truck_id: str = truck_id
        self.refill_df: Optional[pl.DataFrame] = pl.DataFrame()
        self.fuel_supply_df: Optional[pl.DataFrame] = pl.DataFrame()
        self.merged_df: Optional[pl.DataFrame] = pl.DataFrame()

    def load_datasets(self) -> None:
        """
        Carga los datasets de refill y fuel supply desde archivos CSV.

        NOTA: Actualmente las rutas están hardcodeadas temporalmente para desarrollo inicial.
        En el futuro, se debe usar `refill_path` y `fuel_supply_path` como entrada real o implementar
        una obtención desde base de datos.
        """
        refill_path: str = f"refill/{self.truck_id}_refill_events.csv"
        fuel_supply_path: str = f"output/{self.truck_id}_fuel_supply.csv"

        self.refill_df = pl.read_csv(refill_path).with_columns(
            pl.col("TimeStamp").str.strptime(pl.Datetime)
        )
        self.fuel_supply_df = pl.read_csv(fuel_supply_path).with_columns(
            pl.col("TimeStamp").str.strptime(pl.Datetime)
        )

    def filter_by_date_range(self, start_date: date, end_date: date) -> None:
        """Filtra ambos datasets por rango de fechas, funcion opcional que se le puede aplicar."""
        if (
            self.refill_df is None
            or self.fuel_supply_df is None
            or self.refill_df.is_empty()
            or self.fuel_supply_df.is_empty()
        ):
            raise RuntimeError("Debe cargar los datasets primero.")

        date_filter: pl.Expr = (pl.col("TimeStamp").dt.date() >= start_date) & (
            pl.col("TimeStamp").dt.date() <= end_date
        )
        self.refill_df = self.refill_df.filter(date_filter)
        self.fuel_supply_df = self.fuel_supply_df.filter(date_filter)

    def _prepare(self) -> None:
        """Prepara ambos datasets para la correlación."""
        if self.refill_df is None or self.fuel_supply_df is None:
            raise RuntimeError(
                "You must load the datasets first using load_datasets()."
            )

        self.refill_df = (
            self.refill_df.select(
                ["delta_fuel", "TimeStamp", "before_avg", "after_avg"]
            )
            .sort(["TimeStamp"])
            .rename({"TimeStamp": "RefillTimeStamp"})
            .with_columns(
                pl.when(pl.col("RefillTimeStamp").dt.hour().is_between(6, 17))
                .then(pl.lit("D"))
                .otherwise(pl.lit("N"))
                .alias("refill_shift")
            )
        )

        self.fuel_supply_df = (
            self.fuel_supply_df.select(["Origin", "TimeStamp", "FuelLevelLiters"])
            .sort(["TimeStamp"])
            .rename({"TimeStamp": "SupplyTimeStamp"})
            .with_columns(
                pl.when(pl.col("SupplyTimeStamp").dt.hour().is_between(6, 17))
                .then(pl.lit("D"))
                .otherwise(pl.lit("N"))
                .alias("supply_shift")
            )
        )

    def _calculate_optimal_adjustment(self, df: pl.DataFrame) -> Tuple[float, float]:
        """
        Calcula dos valores óptimos para ajustar delta_fuel y minimizar fuel_discrepancy:
        - subtract_value: valor a restar cuando delta_fuel > FuelLevelLiters
        - add_value: valor a sumar cuando delta_fuel < FuelLevelLiters

        Returns:
            Tuple[float, float]: (subtract_value, add_value)
        """
        # Filtrar solo filas con ambos valores presentes
        valid_data: pl.DataFrame = df.filter(
            (pl.col("delta_fuel").is_not_null())
            & (pl.col("FuelLevelLiters").is_not_null())
        )

        if valid_data.height == 0:
            print("Advertencia: No hay datos válidos para optimización")
            return (0.0, 0.0)

        # Convertir a numpy arrays para optimización
        higher_data: pl.DataFrame = valid_data.filter(
            pl.col("delta_fuel") > pl.col("FuelLevelLiters")
        )
        subtract_value = 0.0

        if higher_data.height > 0:

            delta_fuel_higher: ndarray[tuple[int], Any] = (
                higher_data.select("delta_fuel").to_numpy().flatten()
            )
            fuel_level_higher: ndarray[tuple[int], Any] = (
                higher_data.select("FuelLevelLiters").to_numpy().flatten()
            )

            def objective_function_subtract(adjustment_value) -> float:
                """Función objetivo para valores a restar (adjustment_value será positivo)"""
                adjusted_delta_fuel = delta_fuel_higher - abs(
                    adjustment_value
                )  # Siempre restar
                discrepancy: Any = np.abs(adjusted_delta_fuel - fuel_level_higher)
                return float(np.mean(discrepancy))

            # Rango de búsqueda para restar
            max_difference = np.max(delta_fuel_higher - fuel_level_higher)
            search_bounds_subtract = (
                0,
                max_difference * 1.5,
            )  # Solo valores positivos para restar

            result_subtract: Any = minimize_scalar(
                objective_function_subtract,
                bounds=search_bounds_subtract,
                method="bounded",
            )

            if result_subtract.success:
                subtract_value: float = abs(
                    result_subtract.x
                )  # Asegurar que sea positivo
                print(f"Valor óptimo a RESTAR: {subtract_value:.6f}")
                print(
                    f"Discrepancia promedio mínima (restar): {result_subtract.fun:.6f}"
                )
            else:
                print("Advertencia: Optimización para restar no convergió")
        else:
            subtract_value = 0.0

        lower_data: pl.DataFrame = valid_data.filter(
            pl.col("delta_fuel") < pl.col("FuelLevelLiters")
        )
        add_value = 0.0

        if lower_data.height > 0:
            print(
                f"Analizando {lower_data.height} registros donde delta_fuel < FuelLevelLiters"
            )

            delta_fuel_lower: ndarray[tuple[int], Any] = (
                lower_data.select("delta_fuel").to_numpy().flatten()
            )
            fuel_level_lower: ndarray[tuple[int], Any] = (
                lower_data.select("FuelLevelLiters").to_numpy().flatten()
            )

            def objective_function_add(adjustment_value) -> float:
                """Función objetivo para valores a sumar (adjustment_value será positivo)"""
                adjusted_delta_fuel = delta_fuel_lower + abs(
                    adjustment_value
                )  # Siempre sumar
                discrepancy = np.abs(adjusted_delta_fuel - fuel_level_lower)
                return float(np.mean(discrepancy))

            # Rango de búsqueda para sumar
            max_difference = np.max(fuel_level_lower - delta_fuel_lower)
            search_bounds_add = (
                0,
                max_difference * 1.5,
            )  # Solo valores positivos para sumar

            result_add: Any = minimize_scalar(
                objective_function_add, bounds=search_bounds_add, method="bounded"
            )

            if result_add.success:
                add_value = abs(result_add.x)  # Asegurar que sea positivo
                print(f"Valor óptimo a SUMAR: {add_value:.6f}")
                print(f"Discrepancia promedio mínima (sumar): {result_add.fun:.6f}")
            else:
                print("Advertencia: Optimización para sumar no convergió")
        else:
            print("No hay registros donde delta_fuel < FuelLevelLiters")

        equal_data = valid_data.filter(
            pl.col("delta_fuel") == pl.col("FuelLevelLiters")
        )
        if equal_data.height > 0:
            print(
                f"Encontrados {equal_data.height} registros donde delta_fuel == FuelLevelLiters (perfectos)"
            )

        print(f"\nRESUMEN DE OPTIMIZACIÓN:")
        print(
            f"  Valor a restar cuando delta_fuel > FuelLevelLiters: {subtract_value:.6f}"
        )
        print(f"  Valor a sumar cuando delta_fuel < FuelLevelLiters: {add_value:.6f}")

        return (subtract_value, add_value)

    def correlate_events(self) -> pl.DataFrame:
        """Correlaciona eventos usando un sistema de puntuación personalizado."""

        # Separar eventos
        refills: pl.DataFrame = (
            self.refill_df if self.refill_df is not None else pl.DataFrame()
        )
        supplies: pl.DataFrame = (
            self.fuel_supply_df if self.fuel_supply_df is not None else pl.DataFrame()
        )

        # Crear producto cartesiano con ventana temporal de ±15 horas
        cross_join: pl.DataFrame = refills.join(
            supplies, how="cross", suffix="_supply"
        ).filter(
            (
                pl.col("SupplyTimeStamp")
                >= pl.col("RefillTimeStamp") - pl.duration(hours=20)
            )
            & (
                pl.col("SupplyTimeStamp")
                <= pl.col("RefillTimeStamp") + pl.duration(hours=20)
            )
        )
        # Calcular puntuación (ejemplo con 2 criterios)
        scored: pl.DataFrame = cross_join.with_columns(
            # Criterio 1: Proximidad temporal (40%)
            time_score=(
                1
                - (pl.col("RefillTimeStamp") - pl.col("SupplyTimeStamp")).abs()
                / pl.duration(hours=12)
            )
            * 0.40,
            # Criterio 2: Discrepancia de combustible entre delta y recarga combustible (30%)
            fuel_diff_score=(
                1
                - (pl.col("delta_fuel") - pl.col("FuelLevelLiters")).abs()
                / pl.max_horizontal([pl.col("delta_fuel"), pl.col("FuelLevelLiters")])
            )
            * 0.35,
            # Criterio 3: Consideracion del origen de la recarga (20%)
            origin_score=(
                pl.when(pl.col("Origin").is_in(["SURTIDOR-TRUCKSHOP"]))
                .then(0.10)
                .otherwise(0.05)  # 15 % del puntaje total
            ),
            shift_score=(
                pl.when(pl.col("supply_shift") == pl.col("refill_shift"))
                .then(0.10)  # 10% del puntaje total
                .otherwise(0.0)
            ),
        ).with_columns(
            total_score=pl.col("time_score")
            + pl.col("fuel_diff_score")
            + pl.col("origin_score")
            + pl.col("shift_score"),
        )

        # Filtramos por score aceptable y ordenamos por mayor puntuación
        scored_sorted = scored.filter(pl.col("total_score") > 0).sort(
            "total_score", descending=True
        )

        # Creamos sets para evitar emparejamientos repetidos
        matched_refills: set[datetime] = set()
        matched_supplies: set[tuple[datetime, float]] = set()
        best_matches_rows = []

        # Recorremos por orden de puntuación (voraz)
        for row in scored_sorted.iter_rows(named=True):
            refill_ts: Any = row["RefillTimeStamp"]
            supply_ts: tuple[Any, Any] = (
                row["SupplyTimeStamp"],
                row["FuelLevelLiters"],
            )

            if refill_ts in matched_refills or supply_ts in matched_supplies:
                continue

            matched_refills.add(refill_ts)
            matched_supplies.add(supply_ts)
            best_matches_rows.append(row)

        # Convertimos los matches a DataFrame
        best_matches: pl.DataFrame = pl.DataFrame(best_matches_rows)

        # Obtener supplies no emparejados
        matched_supplies = set(
            best_matches.select("SupplyTimeStamp").to_series().to_list()
        )
        unmatched_supplies: pl.DataFrame = supplies.filter(
            ~pl.col("SupplyTimeStamp").is_in(matched_supplies)
        )

        # Obtener refill no emparejados
        matched_refills = set(
            best_matches.select("RefillTimeStamp").to_series().to_list()
        )
        unmatched_refills: pl.DataFrame = refills.filter(
            ~pl.col("RefillTimeStamp").is_in(matched_refills)
        )

        unmatched_supplies = unmatched_supplies.with_columns(
            pl.lit(None).alias("delta_fuel"),
            pl.lit(None).alias("RefillTimeStamp"),
            pl.lit(None).alias("before_avg"),
            pl.lit(None).alias("after_avg"),
            pl.lit(None).alias("refill_shift"),
        )

        unmatched_refills = unmatched_refills.with_columns(
            pl.lit(None).alias("Origin"),
            pl.lit(None).alias("SupplyTimeStamp"),
            pl.lit(None).alias("FuelLevelLiters"),
            pl.lit(None).alias("supply_shift"),
        )

        # Seleccionar mismas columnas en ambos DataFrames
        columns_to_select = [
            "delta_fuel",
            "RefillTimeStamp",
            "before_avg",
            "after_avg",
            "Origin",
            "SupplyTimeStamp",
            "FuelLevelLiters",
            "supply_shift",
            "refill_shift",
        ]

        correlate_df = pl.concat(
            [
                best_matches.select(columns_to_select),
                unmatched_supplies.select(columns_to_select),
                unmatched_refills.select(columns_to_select),
            ]
        ).sort(pl.coalesce("RefillTimeStamp", "SupplyTimeStamp"))

        # Crear DataFrame inicial con métricas básicas
        initial_df = correlate_df.with_columns(
            time_discrepancy=(pl.col("RefillTimeStamp") - pl.col("SupplyTimeStamp"))
            .abs()
            .dt.total_seconds(),
            fuel_discrepancy=((pl.col("delta_fuel")) - pl.col("FuelLevelLiters")).abs(),
        ).with_columns(
            # Clasificación temporal
            pl.when((pl.col("RefillTimeStamp") >= pl.col("SupplyTimeStamp")))
            .then(pl.lit("TSNormal"))
            .otherwise(pl.lit("TSAtrasado"))
            .alias("classification"),
            # Clasificación del tipo de evento
            pl.when(
                (pl.col("RefillTimeStamp").is_not_null())
                & (pl.col("SupplyTimeStamp").is_not_null())
            )
            .then(pl.lit("Both_Events"))
            .when(
                (pl.col("RefillTimeStamp").is_not_null())
                & (pl.col("SupplyTimeStamp").is_null())
            )
            .then(pl.lit("Refill_Only"))
            .when(
                pl.col("RefillTimeStamp").is_null()
                & pl.col("SupplyTimeStamp").is_not_null()
            )
            .then(pl.lit("Supply_Only"))
            .otherwise(pl.lit("Unknown"))
            .alias("event_type"),
        )

        subtract_value, add_value = self._calculate_optimal_adjustment(initial_df)

        self.merged_df = initial_df.with_columns(
            # Aplicar ajuste solo donde delta_fuel es menor a registros
            pl.when(
                (pl.col("delta_fuel").is_not_null())
                & (pl.col("FuelLevelLiters").is_not_null())
                & (pl.col("delta_fuel") < pl.col("FuelLevelLiters"))
            )
            .then(pl.col("delta_fuel") + add_value)
            .when(
                (pl.col("delta_fuel").is_not_null())
                & (pl.col("FuelLevelLiters").is_not_null())
                & (pl.col("delta_fuel") > pl.col("FuelLevelLiters"))
            )
            .then(pl.col("delta_fuel") - subtract_value)
            .otherwise(pl.col("delta_fuel"))
            .alias("delta_fuel")
        ).with_columns(
            # Recalcular fuel_discrepancy con el valor ajustado
            pl.when(
                (pl.col("delta_fuel").is_not_null())
                & (pl.col("FuelLevelLiters").is_not_null())
            )
            .then((pl.col("delta_fuel") - pl.col("FuelLevelLiters")).abs())
            .otherwise(pl.col("fuel_discrepancy"))
            .alias("fuel_discrepancy"),
            pl.lit(self.truck_id).alias("truck_id"),
        )

        return self.merged_df.select(
            [
                "RefillTimeStamp",
                "delta_fuel",
                "before_avg",
                "after_avg",
                "Origin",
                "SupplyTimeStamp",
                "FuelLevelLiters",
                "time_discrepancy",
                "fuel_discrepancy",
                "classification",
                "event_type",
                "truck_id",
            ]
        )

    def get_result(self) -> Optional[pl.DataFrame]:
        """Retorna el DataFrame correlacionado final."""
        return self.merged_df

    def save_result(self, path: Union[str, Path], format: str = "csv"):
        """Guarda el resultado correlacionado en un archivo."""
        if self.merged_df is None:
            raise RuntimeError(
                "No hay datos para guardar. Ejecute primero execute_correlation()."
            )

        if format == "csv":
            self.merged_df.write_csv(path)
        elif format == "parquet":
            self.merged_df.write_parquet(path)
        else:
            raise ValueError(f"Formato '{format}' no soportado. Usa 'csv' o 'parquet'.")

    def get_column_statistics(self, column_name: str) -> Dict[str, Any]:
        """
        Calcula estadísticas detalladas para una columna específica del DataFrame correlacionado.
        Debes haber ejecutado `execute_correlation()` previamente.
        """
        if self.merged_df is None:
            raise RuntimeError(
                "No hay datos cargados. Ejecuta primero `execute_correlation()`."
            )

        df = self.merged_df

        if column_name not in df.columns:
            raise ValueError(f"La columna '{column_name}' no existe en el DataFrame.")

        col_type = df.schema[column_name]
        col = df[column_name]

        # Para columnas de tipo datetime
        if col_type == pl.Datetime:
            min_val = col.min()
            max_val = col.max()
            first_record = (
                min_val.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(min_val, datetime)
                else min_val
            )
            last_record = (
                max_val.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(max_val, datetime)
                else max_val
            )
            total_duration_seconds = (
                (max_val - min_val).total_seconds()
                if isinstance(min_val, datetime) and isinstance(max_val, datetime)
                else None
            )
            return {
                "first_record": first_record,
                "last_record": last_record,
                "total_duration_seconds": total_duration_seconds,
            }

        # Para columnas numéricas
        elif col_type in [pl.Float64, pl.Int64]:
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            return {
                "mean": col.mean(),
                "median": col.median(),
                "mode": col.mode().to_list()[0] if not col.mode().is_empty() else None,
                "min": col.min(),
                "max": col.max(),
                "variance": col.var(),
                "std_dev": col.std(),
                "q1": q1,
                "q3": q3,
                "iqr": float(q3 - q1) if q1 is not None and q3 is not None else None,
                "skewness": col.skew(),
                "kurtosis": col.kurtosis(),
                "non_null_count": col.drop_nulls().len(),
                "null_count": col.null_count(),
            }

        else:
            # Estadísticas para columnas categóricas o tipo string
            return {
                "unique_values": col.n_unique(),
                "most_frequent": (
                    col.mode().to_list()[0] if not col.mode().is_empty() else None
                ),
                "non_null_count": col.drop_nulls().len(),
                "null_count": col.null_count(),
            }


import os

if __name__ == "__main__":
    # Lista de camiones
    TRUCK_SPECS = [
        "T-210",
        "T-211",
        "T-212",
        "T-213",
        "T-214",
        "T-215",
        "T-216",
        "T-217",
        "T-218",
        "T-219",
        "T-220",
        "T-221",
        "T-222",
        "T-223",
        "T-224",
        "T-225",
        "T-230",
        "T-231",
        "T-232",
        "T-233",
        "T-234",
        "T-235",
        "T-236",
        "T-237",
        "T-238",
        "T-239",
        "T-240",
        "T-241",
        "T-242",
        "T-243",
    ]
    # Crear directorio para resultados
    os.makedirs("correlated_events", exist_ok=True)

    # Rango de fechas fijo para todos los camiones
    start_date = date(2024, 2, 1)
    end_date = date(2025, 2, 28)

    print("=" * 60)
    print(f"🔗 Correlacionando eventos para {len(TRUCK_SPECS)} camiones")
    print(f"📅 Rango de fechas: {start_date} a {end_date}")
    print("=" * 60)

    result: pl.DataFrame = pl.DataFrame()

    for truck_id in TRUCK_SPECS:
        print(f"\n🚚 Procesando camión {truck_id}...")

        # 1. Inicializar correlator para el camión
        correlator = SensorSupplyEventCorrelator(truck_id)

        try:
            # 2. Cargar datos
            correlator.load_datasets()

            # 3. Filtrar por rango de fechas
            correlator.filter_by_date_range(start_date, end_date)

            # 4. Preparar datos
            correlator._prepare()

            # 5. Correlacionar eventos
            df = correlator.correlate_events()
            result = pl.concat([result, df])

            print(f"📦 Columnas: {result.columns}")

        except Exception as e:
            print(f"❌ Error procesando {truck_id}: {str(e)}")
            # 6. Guardar resultado

    output_path = f"correlated_events/all_correlated_events.csv"
    result.write_csv(output_path)
    print("\n🎉 Proceso completado para todos los camiones!")
