import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import date, timedelta


class SensorSupplyEventCorrelator:
    """
    Correlaciona eventos de refill (sensores) con eventos de fuel supply
    bajo criterios específicos de agrupación temporal y emparejamiento.
    """

    def __init__(self):
        self.refill_df: Optional[pl.DataFrame] = None
        self.fuel_supply_df: Optional[pl.DataFrame] = None
        self.merged_df: Optional[pl.DataFrame] = None

    def load_datasets(self):
        """
        Carga los datasets de refill y fuel supply desde archivos CSV.

        NOTA: Actualmente las rutas están hardcodeadas temporalmente para desarrollo inicial.
        En el futuro, se debe usar `refill_path` y `fuel_supply_path` como entrada real o implementar
        una obtención desde base de datos.
        """

        self.refill_df = pl.read_csv("refill_events.csv").with_columns(
            pl.col("TimeStamp").str.strptime(pl.Datetime)
        )
        self.fuel_supply_df = pl.read_csv("output/T-210_fuel_supply.csv").with_columns(
            pl.col("TimeStamp").str.strptime(pl.Datetime)
        )

    def filter_by_date_range(self, start_date: date, end_date: date):
        """Filtra ambos datasets por rango de fechas, funcion opcional que se le puede aplicar."""
        if self.refill_df.is_empty() or self.fuel_supply_df.is_empty():
            raise RuntimeError("Debe cargar los datasets primero.")

        date_filter = (pl.col("TimeStamp").dt.date() >= start_date) & (
            pl.col("TimeStamp").dt.date() <= end_date
        )
        self.refill_df = self.refill_df.filter(date_filter)
        self.fuel_supply_df = self.fuel_supply_df.filter(date_filter)

    def _prepare(self):
        """Prepara ambos datasets para la correlación."""
        self.refill_df = (
            self.refill_df.select(["DeltaFuel", "TimeStamp", "before_avg", "after_avg"])
            .sort(["TimeStamp"])
            .rename({"TimeStamp": "RefillTimeStamp"})
        )

        self.fuel_supply_df = (
            self.fuel_supply_df.select(["Origin", "TimeStamp", "FuelLevelLiters"])
            .sort(["TimeStamp"])
            .rename({"TimeStamp": "SupplyTimeStamp"})
        )

    def correlate_events(self) -> pl.DataFrame:
        """Correlaciona eventos usando un sistema de puntuación personalizado."""

        # Separar eventos
        refills = self.refill_df
        supplies = self.fuel_supply_df

        # Crear producto cartesiano con ventana temporal de ±12 horas
        cross_join = refills.join(supplies, how="cross", suffix="_supply").filter(
            (
                pl.col("SupplyTimeStamp")
                >= pl.col("RefillTimeStamp") - pl.duration(hours=12)
            )
            & (
                pl.col("SupplyTimeStamp")
                <= pl.col("RefillTimeStamp") + pl.duration(hours=12)
            )
        )
        # Calcular puntuación (ejemplo con 2 criterios)
        scored = cross_join.with_columns(
            # Criterio 1: Proximidad temporal (50%)
            time_score=(
                1
                - (pl.col("RefillTimeStamp") - pl.col("SupplyTimeStamp")).abs()
                / pl.duration(hours=12)
            )
            * 0.5,
            # Criterio 2: Discrepancia de combustible entre delta y recarga combustible
            fuel_diff_score=(
                1
                - (pl.col("DeltaFuel") - pl.col("FuelLevelLiters")).abs()
                / pl.max_horizontal([pl.col("DeltaFuel"), pl.col("FuelLevelLiters")])
            )
            * 0.3,
            # Criterio 3: Consideracion del origen de la recarga
            origin_score=(
                pl.when(pl.col("Origin").is_in(["SURTIDOR-TRUCKSHOP"]))
                .then(70)
                .otherwise(30)
            )
            * 0.2,
        ).with_columns(
            total_score=pl.col("time_score")
            + pl.col("fuel_diff_score")
            + pl.col("origin_score"),
        )

        best_matches = (
            scored.filter(pl.col("total_score") > 0)
            .sort("total_score", descending=True)
            .group_by("RefillTimeStamp")
            .first()
        )

        # Obtener supplies no emparejados
        matched_supplies = best_matches.select("SupplyTimeStamp").to_series().implode()
        unmatched_supplies = supplies.filter(
            ~pl.col("SupplyTimeStamp").is_in(matched_supplies)
        )

        # Obtener refill no emparejados
        matched_refills = best_matches.select("RefillTimeStamp").to_series().implode()
        unmatched_refills = refills.filter(
            ~pl.col("RefillTimeStamp").is_in(matched_refills)
        )

        unmatched_supplies = unmatched_supplies.with_columns(
            pl.lit(None).alias("DeltaFuel"),
            pl.lit(None).alias("RefillTimeStamp"),
            pl.lit(None).alias("before_avg"),
            pl.lit(None).alias("after_avg"),
        )

        unmatched_refills = unmatched_refills.with_columns(
            pl.lit(None).alias("Origin"),
            pl.lit(None).alias("SupplyTimeStamp"),
            pl.lit(None).alias("FuelLevelLiters"),
        )

        # Seleccionar mismas columnas en ambos DataFrames
        columns_to_select = [
            "DeltaFuel",
            "RefillTimeStamp",
            "before_avg",
            "after_avg",
            "Origin",
            "SupplyTimeStamp",
            "FuelLevelLiters",
        ]

        correlate_df = pl.concat(
            [
                best_matches.select(columns_to_select),
                unmatched_supplies.select(columns_to_select),
                unmatched_refills.select(columns_to_select),
            ]
        ).sort(pl.coalesce("RefillTimeStamp", "SupplyTimeStamp"))

        self.merged_df = correlate_df.with_columns(
            time_discrepancy=(pl.col("RefillTimeStamp") - pl.col("SupplyTimeStamp"))
            .abs()
            .dt.total_seconds(),
            fuel_discrepancy=((pl.col("DeltaFuel")) - pl.col("FuelLevelLiters")).abs(),
            supply_turno=pl.when(pl.col("SupplyTimeStamp").dt.hour().is_between(6, 17))
            .then(pl.lit("D"))
            .otherwise(pl.lit("N")),
            refill_turno=pl.when(pl.col("RefillTimeStamp").dt.hour().is_between(6, 17))
            .then(pl.lit("D"))
            .otherwise(pl.lit("N")),
        ).with_columns(
            classification=pl.when(
                (pl.col("RefillTimeStamp") >= pl.col("SupplyTimeStamp"))
            )
            .then(pl.lit("TSNormal"))
            .otherwise(pl.lit("TSAtrasado")),
            event_type=pl.when(
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
            .otherwise(pl.lit("Unknown")),
        )

        return self.merged_df

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
            return {
                "first_record": col.min().strftime("%Y-%m-%d %H:%M:%S"),
                "last_record": col.max().strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": (col.max() - col.min()).total_seconds(),
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


from datetime import date

correlator = SensorSupplyEventCorrelator()
correlator.load_datasets()
correlator.filter_by_date_range(date(2024, 2, 1), date(2025, 2, 27))
correlator._prepare()
# result = correlator._create_unified_timeline()
result = correlator.correlate_events()
correlator.save_result("correlated_events3.csv", format="csv")
print(result.height)
print(result.head(10))
