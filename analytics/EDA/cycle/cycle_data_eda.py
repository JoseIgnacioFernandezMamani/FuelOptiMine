import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
from etl_core.load.utils import create_client, CH_CONFIG


class CycleDataEDA:
    """
    Análisis exploratorio de datos de ciclos mineros.
    Enfocado en variables categóricas y análisis temporal.
    """

    def __init__(self, truck_id: Optional[str] = None) -> None:
        self.cycle_df: pl.DataFrame = pl.DataFrame()
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        self._data_loaded: bool = False
        self._stats_generated: bool = False
        self.truck_id = truck_id
        self.client = create_client(CH_CONFIG)

        # Definición de columnas
        self.categorical_cols = [
            "Shovel",
            "ShovelModel",
            "StageType",
            "LoadingZone",
            "Material",
            "DestinationType",
            "Destination",
        ]

        self.numeric_cols = [
            "StageSequence",
            "MeasuredTonnage",
            "ReportedTonnage",
            "Distance",
            "TimeEfficiencyPercentage",
        ]

        self.datetime_cols = ["TimeStampIni", "TimeStampFin"]

    def _load_cycle_data(self):
        """Cargar datos de ciclos desde ClickHouse"""
        try:
            # Query base - ajustar según tu tabla real
            if self.truck_id:
                query = f"""
                SELECT 
                    CycleId,
                    Shovel,
                    ShovelModel,
                    StageType,
                    StageSequence,
                    TimeStampIni,
                    TimeStampFin,
                    LoadingZone,
                    Material,
                    MeasuredTonnage,
                    ReportedTonnage,
                    DestinationType,
                    Destination,
                    Distance,
                    TimeEfficiencyPercentage
                FROM cycles_table
                WHERE Equipment = '{self.truck_id}'
                ORDER BY TimeStampIni
                """
            else:
                query = """
                SELECT * FROM cycles_table
                ORDER BY TimeStampIni
                """

            pandas_df = self.client.query_df(query)

            if pandas_df.empty:
                raise RuntimeError("No se encontraron datos de ciclos")

            self.cycle_df = pl.from_pandas(pandas_df)

            # Calcular columnas derivadas
            self.cycle_df = self.cycle_df.with_columns(
                [
                    # Duración del ciclo en segundos
                    (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
                    .dt.total_seconds()
                    .alias("CycleDurationSeconds"),
                    # Diferencia entre tonelaje medido y reportado
                    (pl.col("MeasuredTonnage") - pl.col("ReportedTonnage")).alias(
                        "TonnageDifference"
                    ),
                    # Porcentaje de error en tonelaje
                    (
                        (pl.col("MeasuredTonnage") - pl.col("ReportedTonnage"))
                        / pl.col("ReportedTonnage")
                        * 100
                    ).alias("TonnageErrorPercent"),
                    # Extraer componentes temporales
                    pl.col("TimeStampIni").dt.hour().alias("Hour"),
                    pl.col("TimeStampIni").dt.date().alias("Date"),
                    pl.col("TimeStampIni").dt.month().alias("Month"),
                    pl.col("TimeStampIni").dt.weekday().alias("Weekday"),
                ]
            )

            # Actualizar listas de columnas
            self.numeric_cols.extend(
                ["CycleDurationSeconds", "TonnageDifference", "TonnageErrorPercent"]
            )

            self._data_loaded = True
            self._stats_cache = {}
            self._stats_generated = False

        except Exception as e:
            raise RuntimeError(f"Error al cargar datos de ciclos: {str(e)}")

    def _generate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Generar estadísticas para variables categóricas y numéricas"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        stats = {}
        df = self.cycle_df

        for col in df.columns:
            col_type = df.schema[col]
            col_stats = {}

            # Variables categóricas
            if col in self.categorical_cols or col_type == pl.Utf8:
                value_counts = df[col].value_counts().sort("count", descending=True)
                unique_count = df[col].n_unique()

                col_stats = {
                    "type": "categorical",
                    "unique_values": unique_count,
                    "most_common": value_counts.head(5).to_dicts(),
                    "null_count": df[col].null_count(),
                    "null_percentage": (df[col].null_count() / len(df)) * 100,
                }

            # Variables numéricas
            elif col_type in [pl.Float64, pl.Int64] and col not in ["CycleId"]:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)

                col_stats = {
                    "type": "numeric",
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "std_dev": df[col].std(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": float(q3 - q1) if q1 and q3 else None,
                    "p5": df[col].quantile(0.05),
                    "p95": df[col].quantile(0.95),
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "non_null_count": df[col].len(),
                    "null_count": df[col].null_count(),
                    "cv": (
                        (df[col].std() / df[col].mean())
                        if df[col].mean() != 0
                        else None
                    ),
                }

            # Variables datetime
            elif col_type in [pl.Datetime]:
                min_date = df[col].min()
                max_date = df[col].max()

                col_stats = {
                    "type": "datetime",
                    "first_record": (
                        min_date.strftime("%Y-%m-%d %H:%M:%S") if min_date else None
                    ),
                    "last_record": (
                        max_date.strftime("%Y-%m-%d %H:%M:%S") if max_date else None
                    ),
                    "total_duration_days": (
                        ((max_date - min_date).total_seconds() / 86400)
                        if min_date and max_date
                        else None
                    ),
                }

            stats[col] = col_stats

        self._stats_generated = True
        self._stats_cache = stats
        return stats

    def get_categorical_analysis(self, column: str) -> Dict[str, Any]:
        """
        Análisis detallado de una variable categórica.
        Incluye frecuencias, proporciones y relaciones.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        if column not in self.categorical_cols:
            raise ValueError(f"'{column}' no es una columna categórica")

        df = self.cycle_df

        # Frecuencias absolutas y relativas
        value_counts = df[column].value_counts().sort("count", descending=True)
        total = len(df)

        value_counts = value_counts.with_columns(
            [(pl.col("count") / total * 100).alias("percentage")]
        )

        # Entropía (medida de diversidad)
        probs = value_counts["percentage"] / 100
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)

        return {
            "column": column,
            "unique_values": df[column].n_unique(),
            "total_records": total,
            "value_counts": value_counts.to_dicts(),
            "entropy": entropy,
            "max_entropy": np.log2(df[column].n_unique()),
            "diversity_ratio": (
                entropy / np.log2(df[column].n_unique())
                if df[column].n_unique() > 1
                else 0
            ),
        }

    def get_crosstab_analysis(
        self, row_col: str, col_col: str, normalize: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Tabla de contingencia entre dos variables categóricas.

        Args:
            row_col: Variable para filas
            col_col: Variable para columnas
            normalize: 'index', 'columns', 'all' o None
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        crosstab = (
            self.cycle_df.group_by([row_col, col_col])
            .agg(pl.count().alias("count"))
            .pivot(index=row_col, columns=col_col, values="count")
            .fill_null(0)
        )

        if normalize == "index":
            # Normalizar por filas
            numeric_cols = [c for c in crosstab.columns if c != row_col]
            row_sums = crosstab.select(numeric_cols).sum_horizontal()

            for col in numeric_cols:
                crosstab = crosstab.with_columns(
                    (pl.col(col) / row_sums * 100).alias(col)
                )

        elif normalize == "columns":
            # Normalizar por columnas
            numeric_cols = [c for c in crosstab.columns if c != row_col]

            for col in numeric_cols:
                col_sum = crosstab[col].sum()
                crosstab = crosstab.with_columns(
                    (pl.col(col) / col_sum * 100).alias(col)
                )

        elif normalize == "all":
            # Normalizar por total
            numeric_cols = [c for c in crosstab.columns if c != row_col]
            total = crosstab.select(numeric_cols).sum().sum_horizontal()[0]

            for col in numeric_cols:
                crosstab = crosstab.with_columns((pl.col(col) / total * 100).alias(col))

        return crosstab

    def get_numeric_by_category(
        self, numeric_col: str, category_col: str
    ) -> pl.DataFrame:
        """
        Análisis de variable numérica agrupada por categoría.
        Útil para comparar distribuciones.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        summary = (
            self.cycle_df.group_by(category_col)
            .agg(
                [
                    pl.count().alias("count"),
                    pl.col(numeric_col).mean().alias("mean"),
                    pl.col(numeric_col).median().alias("median"),
                    pl.col(numeric_col).std().alias("std_dev"),
                    pl.col(numeric_col).min().alias("min"),
                    pl.col(numeric_col).max().alias("max"),
                    pl.col(numeric_col).quantile(0.25).alias("q1"),
                    pl.col(numeric_col).quantile(0.75).alias("q3"),
                ]
            )
            .sort("mean", descending=True)
        )

        return summary

    def get_temporal_patterns(
        self, group_by: str = "hour", metric: str = "count"
    ) -> pl.DataFrame:
        """
        Analizar patrones temporales (por hora, día, mes).

        Args:
            group_by: 'hour', 'weekday', 'month', 'date'
            metric: 'count', 'avg_tonnage', 'avg_duration', 'avg_efficiency'
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        # Mapeo de agrupación
        time_mapping = {
            "hour": "Hour",
            "weekday": "Weekday",
            "month": "Month",
            "date": "Date",
        }

        group_col = time_mapping.get(group_by, "Hour")

        # Definir agregaciones
        aggs = [pl.count().alias("cycle_count")]

        if metric in ["avg_tonnage", "all"]:
            aggs.append(pl.col("MeasuredTonnage").mean().alias("avg_tonnage"))

        if metric in ["avg_duration", "all"]:
            aggs.append(pl.col("CycleDurationSeconds").mean().alias("avg_duration"))

        if metric in ["avg_efficiency", "all"]:
            aggs.append(
                pl.col("TimeEfficiencyPercentage").mean().alias("avg_efficiency")
            )

        result = self.cycle_df.group_by(group_col).agg(aggs).sort(group_col)

        return result

    def get_efficiency_analysis(self) -> Dict[str, Any]:
        """
        Análisis específico de eficiencia temporal.
        """
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")

        df = self.cycle_df

        # Clasificar eficiencia
        efficiency_ranges = df.with_columns(
            [
                pl.when(pl.col("TimeEfficiencyPercentage") >= 90)
                .then(pl.lit("Excelente (≥90%)"))
                .when(pl.col("TimeEfficiencyPercentage") >= 75)
                .then(pl.lit("Buena (75-90%)"))
                .when(pl.col("TimeEfficiencyPercentage") >= 60)
                .then(pl.lit("Regular (60-75%)"))
                .otherwise(pl.lit("Baja (<60%)"))
                .alias("EfficiencyClass")
            ]
        )

        class_counts = efficiency_ranges["EfficiencyClass"].value_counts()

        # Eficiencia por pala
        by_shovel = (
            df.group_by("Shovel")
            .agg(
                [
                    pl.count().alias("cycles"),
                    pl.col("TimeEfficiencyPercentage").mean().alias("avg_efficiency"),
                    pl.col("CycleDurationSeconds").mean().alias("avg_duration"),
                ]
            )
            .sort("avg_efficiency", descending=True)
        )

        return {
            "overall_avg_efficiency": df["TimeEfficiencyPercentage"].mean(),
            "overall_median_efficiency": df["TimeEfficiencyPercentage"].median(),
            "efficiency_distribution": class_counts.to_dicts(),
            "efficiency_by_shovel": by_shovel.to_dicts(),
            "low_efficiency_cycles": len(
                df.filter(pl.col("TimeEfficiencyPercentage") < 60)
            ),
        }

    def run(self):
        """Ejecutar análisis completo"""
        self._load_cycle_data()
        self._generate_statistics()

    def get_dataframe(self) -> pl.DataFrame:
        """Obtener DataFrame cargado"""
        if not self._data_loaded:
            raise RuntimeError("Primero ejecute run()")
        return self.cycle_df

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Obtener estadísticas generadas"""
        if not self._stats_generated:
            raise RuntimeError("Primero ejecute run()")
        return self._stats_cache

    def close(self):
        """Cerrar conexión a ClickHouse"""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception as e:
                logging.warning(f"Error cerrando conexión: {e}")


if __name__ == "__main__":
    # Ejemplo de uso
    eda = CycleDataEDA(truck_id="T-243")
    eda.run()

    df = eda.get_dataframe()
    stats = eda.get_statistics()

    print("Estadísticas generales:")
    print(stats)

    print("\nAnálisis de Material:")
    material_analysis = eda.get_categorical_analysis("Material")
    print(material_analysis)

    print("\nTonelaje por Pala:")
    tonnage_by_shovel = eda.get_numeric_by_category("MeasuredTonnage", "Shovel")
    print(tonnage_by_shovel)

    print("\nPatrones por hora:")
    hourly_patterns = eda.get_temporal_patterns(group_by="hour", metric="all")
    print(hourly_patterns)

    print("\nAnálisis de eficiencia:")
    efficiency = eda.get_efficiency_analysis()
    print(efficiency)

    eda.close()
