"""
Antes de apagar el motor, el equipo funcionará baja en vacío durante un periodo de 2 minutos.
Mientras, el Operador debe ingresar la opción de carguío combustible en el sistema Hexagon:
ESTADO > DEMORA > ABASTECIMIENTO DE COMBUSTIBLE

"""

    def _preprocess_sensor_data(self) -> None:
        """Preprocesamiento eficiente con Polars"""

        MAX_REFUEL_TIME = (
            1800  # 30 minutos (tiempo máximo esperado para recargar 3200L)
        )
        MIN_REFUEL_VOLUME = 1000  # Mínimo de litros para considerar recarga

        self.sensor_df = self.sensor_df.with_columns(
            # Calcular promedios móviles
            pl.col("FuelLevelLiters")
            .rolling_mean(window_size=10)
            .alias("AverageBefore"),
            pl.col("FuelLevelLiters")
            .reverse()
            .rolling_mean(window_size=10)
            .alias("AverageAfter"),
            pl.when(    
                (pl.col("FuelLevelLiters") > 0)
                & (pl.col("FuelLevelLiters").shift(1) == 0)
                & (pl.col("OutageDuration") >= 300)
            )
            .then(pl.lit("REFUEL"))
            .when(
                (pl.col("FuelLevelLiters") == 0)
                & (pl.col("RecordDuration") < 180)  # < 3 minutos
                & (pl.col("FuelLevelLiters").shift(-1) > 0)
                & (pl.col("FuelLevelLiters").shift(1) > 0)
            )
            .then("SENSOR_OFF")
            .otherwise(pl.lit("OPERATIONAL"))
            .alias("OperationState"),
        )

        self.refill_threshold = self._calculate_dynamic_threshold()

    def _calculate_dynamic_threshold(self) -> float:
        """Calcula umbral dinámico mejorado para detección de recargas"""
        # Filtrar solo deltas positivos (aumentos de combustible)

        registros_combustible_validos = 373

        """detectar los deltas fuel que no son por sensor apagado"""
        positive_deltas = self.sensor_df.filter(
            (pl.col("DeltaFuel") > 0)
            & (pl.col("SensorOff").shift(1))
            & (~pl.col("SensorOff"))
            & (pl.col("RecordDuration").shift(1) < 300)
            & (pl.col("Speed") <= 1)
            & (pl.col("RPM") <= 1)
        )["DeltaFuel"]

        # Calcular métricas clave
        stats = {
            "q90": positive_deltas.quantile(0.90),
            "median_high": positive_deltas.quantile(0.75),
            "mean_positive": positive_deltas.mean(),
            "std_dev": positive_deltas.std(),
            "min_physical": 800,  # Mínimo físico según conocimiento del dominio
        }

        # Lógica combinada para threshold dinámico
        dynamic_threshold = max(
            stats["min_physical"],
            stats["q90"] * 0.8,  # Percentil 90 ajustado
            stats["median_high"] * 1.5,  # Q3 amplificado
            (stats["mean_positive"] + stats["std_dev"] * 2),  # Media + 2σ
        )

        # Asegurar que no exceda percentil 99.5 para evitar outliers
        return min(dynamic_threshold, positive_deltas.quantile(0.95))

    def _detect_refill_events(self):
        """Detecta eventos de recarga válidos"""
        conditions = (self.sensor_df["DeltaFuel"] > self.refill_threshold) & (
            self.sensor_df["PromedioDespues"] > (self.sensor_df["PromedioAntes"] * 1.5)
        )

        self.sensor_df["RefillValido"] = np.where(
            conditions, self.sensor_df["DeltaFuel"], 0
        )

    def _merge(self) -> pl.DataFrame:
        """Combina datos de sensor con registros manuales"""
        # Filtrar eventos válidos
        peaks = self.sensor_df[self.sensor_df["RefillValido"] > 0][
            ["FullDateTime", "DeltaFuel"]
        ]

        # Merge con registros TRUCKSHOP (12h ventana)
        truckshop_logs = self.fuel_df[
            self.fuel_df["Origin"].str.contains("TRUCKSHOP", na=False)
        ]
        merged_truckshop = pl.merge_asof(
            peaks,
            truckshop_logs[["FullDateTime", "FuelLevelLiters", "Origin"]],
            on="FullDateTime",
            direction="nearest",
            tolerance=pl.Timedelta("12h"),
        )

        # Merge con otros orígenes (24h ventana)
        otros_logs = self.fuel_df[
            ~self.fuel_df["Origin"].str.contains("TRUCKSHOP", na=False)
        ]
        merged_otros = pl.merge_asof(
            peaks,
            otros_logs[["FullDateTime", "FuelLevelLiters", "Origin"]],
            on="FullDateTime",
            direction="forward",
            tolerance=pl.Timedelta("24h"),
        )
        # Combinar resultados
        combined = merged_truckshop.combine_first(merged_otros)

        # Calcular diferencias
        combined["DiferenciaHoras"] = (
            combined["FullDateTime"] - combined["FullDateTime_y"]
        ).dt.total_seconds() / 3600
        combined["DiferenciaLitros"] = abs(
            combined["DeltaFuel"] - combined["FuelLevelLiters"]
        )

        return combined

    def analyze(self) -> pl.DataFrame:
        """Ejecuta todo el pipeline de análisis"""
        self._calculate_metrics()
        self._detect_refill_events()
        raw_results = self._merge_with_manual_logs()

        # Clasificar anomalías
        raw_results["Tipo_Anomalia"] = np.select(
            [
                raw_results["FuelLevelLiters"].isna(),
                raw_results["DiferenciaLitros"] > 500,
                raw_results["DiferenciaHoras"] > 24,
            ],
            [
                "Recarga no registrada",
                "Discrepancia significativa",
                "Registro fuera de ventana",
            ],
            default="Registro válido",
        )

        # Formatear resultados finales
        self.analysis_results = raw_results[
            [
                "FullDateTime",
                "DeltaFuel",
                "FuelLevelLiters",
                "Origin",
                "Tipo_Anomalia",
                "DiferenciaHoras",
                "DiferenciaLitros",
            ]
        ].rename(
            columns={
                "FullDateTime": "FechaHora",
                "DeltaFuel": "LitrosDetectados",
                "FuelLevelLiters": "LitrosRegistrados",
                "Origin": "OrigenRegistro",
            }
        )

        return self.analysis_results


def main():
    # Inicializar analizador
    analyzer = FuelAnalysisOptimized()

    # Ejecutar preprocesamiento
    analyzer._preprocess_sensor_data()

    # Mostrar metadatos básicos
    print("\n" + "=" * 50)
    print("📊 Análisis de Combustible - Visualización en Terminal")
    print("=" * 50)

    # 1. Mostrar estructura de datos del sensor
    print("\n🔧 Datos del Sensor (Estructura):")
    print(f"- Registros: {analyzer.sensor_df.height}")
    print(f"- Columnas: {analyzer.sensor_df.columns}")
    print("- Muestra de datos:")
    print(analyzer.sensor_df.head(3).to_pandas().to_markdown(index=False))

    # 2. Estadísticas clave del sensor
    print("\n📈 Estadísticas del Sensor:")
    stats = analyzer.sensor_df.select(
        [
            pl.col("DeltaFuel").mean().alias("Promedio Delta"),
            pl.col("DeltaFuel").max().alias("Máximo Delta"),
            pl.col("SensorOff").sum().alias("Eventos Apagado"),
        ]
    )
    print(stats.to_pandas().to_markdown(index=False))

    # 3. Umbral de recarga dinámico
    print(f"\n⚖️ Umbral de Recarga Dinámico: {analyzer.refill_threshold:.2f} litros")

    # 4. Detalles de fuel supply
    print("\n⛽ Datos de Suministro de Combustible:")
    print(f"- Registros: {analyzer.fuel_df.height}")
    print("- Muestra de eventos de recarga:")
    print(
        analyzer.fuel_df.filter(pl.col("DeltaFuel") > 0)
        .head(3)
        .to_pandas()
        .to_markdown(index=False)
    )


if __name__ == "__main__":
    main()

























import polars as pl
import numpy as np
from pathlib import Path
from typing import Tuple


class FuelAnalysisOptimized:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent

    def _load_sensor_data(self) -> pl.DataFrame:
        """Carga y prepara datos del sensor"""
        return (
            pl.read_csv(
                self.base_dir / "output" / "T-210_sensor.csv",
                columns=["TimeStamp", "RecordDuration", "FuelLevelLiters"],
            )
            .sort("TimeStamp")
            .with_columns(
                pl.col("FuelLevelLiters").diff().alias("DeltaFuel"),
            )
        )

    def _load_fuel_supply_data(self) -> pl.DataFrame:
        """Carga y prepara datos de combustible"""

        return (
            pl.read_csv(
                self.base_dir / "output" / "T-210_fuel_supply.csv",
                columns=["Origin", "TimeStamp", "LastRefuel", "FuelLevelLiters"],
            )
            .sort("TimeStamp")
            .with_columns(
                pl.col("FuelLevelLiters").diff().alias("DeltaFuel"),
            )
        )

    def _calculate_base_parameters(self) -> None:
        """Calcula parámetros dinámicos sin valores por defecto"""
        # 1. Identificar todos los posibles periodos de apagado
        outage_periods = (
            self.sensor_df.with_columns(
                outage_group=(pl.col("FuelLevelLiters") != 0).cast(int).cum_sum()
            )
            .filter(pl.col("FuelLevelLiters") == 0)
            .group_by("outage_group")
            .agg(
                pl.col("RecordDuration").sum().alias("outage_duration"),
                pl.col("DeltaFuel").shift(-1).fill_null(0).first().alias("post_delta"),
            )
        )

        # 2. Filtrar candidatos a recarga (post_delta > 0)
        refuel_candidates = outage_periods.filter(pl.col("post_delta") > 0)

        # 3. Calcular parámetros usando percentiles
        if refuel_candidates.height > 0:
            # Estadísticas de tiempo
            time_stats = refuel_candidates.select(
                [
                    pl.col("outage_duration").quantile(0.10).alias("p10_time"),
                    pl.col("outage_duration").quantile(0.90).alias("p90_time"),
                ]
            ).to_dicts()[0]

            # Estadísticas de volumen
            volume_stats = refuel_candidates.select(
                [
                    pl.col("post_delta").quantile(0.25).alias("p25_volume"),
                    pl.col("post_delta").quantile(0.75).alias("p75_volume"),
                ]
            ).to_dicts()[0]

            self.min_refuel_time = time_stats["p10_time"]  # Percentil 10 de tiempos
            self.max_refuel_time = time_stats["p90_time"]  # Percentil 90 de tiempos
            self.refill_threshold = volume_stats[
                "p25_volume"
            ]  # Percentil 25 de volumen

        else:
            # Extraer parámetros de distribución completa (sin filtros)
            all_deltas = self.sensor_df.filter(pl.col("DeltaFuel") > 0)["DeltaFuel"]
            all_outages = outage_periods["outage_duration"]

            self.min_refuel_time = (
                all_outages.quantile(0.10) if not all_outages.is_empty() else 0
            )
            self.max_refuel_time = (
                all_outages.quantile(0.90) if not all_outages.is_empty() else 0
            )
            self.refill_threshold = (
                all_deltas.quantile(0.10) if not all_deltas.is_empty() else 0
            )

    def _full_preprocessing(self) -> None:
        """Preprocesamiento completo usando los parámetros calculados"""
        self.sensor_df = self.sensor_df.with_columns(
            # Identificar recargas válidas con parámetros dinámicos
            pl.when(
                (pl.col("DeltaFuel") >= self.refill_threshold)
                & (pl.col("FuelLevelLiters").shift(1) == 0)
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("ValidRefuel"),
            # Calcular duración de apagados relevantes
            pl.when(pl.col("FuelLevelLiters") == 0)
            .then(pl.col("RecordDuration"))
            .otherwise(0)
            .cum_sum()
            .over((pl.col("FuelLevelLiters") == 0).cast(int))
            .alias("OutageDuration"),
        )

    def get_parameters(self) -> dict:
        return {
            "min_refuel_time": self.min_refuel_time,
            "max_refuel_time": self.max_refuel_time,
            "refill_threshold": self.refill_threshold,
        }


if __name__ == "__main__":
    analyzer = FuelAnalysisOptimized()
    params = analyzer.get_parameters()

    print("=== Parámetros Calculados Exclusivamente desde Datos ===")
    print(f"Tiempo mínimo recarga: {params['min_refuel_time']:.1f}s")
    print(f"Tiempo máximo recarga: {params['max_refuel_time']:.1f}s")
    print(f"Volumen mínimo recarga: {params['refill_threshold']:.1f}L")
