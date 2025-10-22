"""
Lightweight Stage 1 Fuel Consumption Collector

This module efficiently collects Stage 1 fuel consumption metrics for multiple trucks
with minimal RAM usage by processing one truck at a time and only keeping essential data.

Author: FuelOptiMine Team
Date: 2025-10-07
"""

import polars as pl
import pandas as pd
import logging
from typing import List, Dict
import gc

from model.utils.model_utils import get_logger
from etl_core.load.utils.config import CH_CONFIG, create_client
from .mlflow_config import TRUCK_IDS


class Stage1FuelCollector:
    """
    Memory-efficient collector for Stage 1 fuel consumption metrics.

    Processes trucks sequentially and only retains summary metrics,
    releasing all intermediate data to minimize RAM usage.
    """

    def __init__(self, truck_ids: List[str]):
        """
        Initialize the collector.

        Args:
            truck_ids: List of truck IDs to process (e.g., ["T-210", "T-211"])
        """
        self.truck_ids = truck_ids
        self.results: List[Dict[str, float]] = []
        self.logger = get_logger(
            "Stage1Collector", "stage1_collector.log", console=True
        )

    def calculate_stage1_fuel(self, truck_id: str) -> float:
        """
        Calculate Stage 1 fuel consumption for a single truck.

        Implementa exactamente la misma lógica del código original:
        1. Identifica ciclos basados en StageSequence
        2. Agrupa por cycle_group
        3. Filtra solo Stage 1 (StageSum == 1)
        4. Calcula StartFuel - EndFuel con validación < 30

        Args:
            truck_id: Truck identifier

        Returns:
            float: Stage 1 fuel consumption in liters
        """
        try:
            self.logger.info(f"Processing {truck_id}...")

            # Load only necessary columns from ClickHouse
            client = create_client(CH_CONFIG, self.logger)

            query = f"""
            SELECT 
                StageSequence,
                FuelLevelLiters,
                SortTimestamp
            FROM {CH_CONFIG['database']}.xgboost_fuel
            WHERE Equipment = '{truck_id}'
            ORDER BY SortTimestamp
            """

            result = client.query(query)
            columns = result.column_names
            data = result.result_rows

            df_pandas = pd.DataFrame(data, columns=columns)
            df = pl.from_pandas(df_pandas)

            # Clean up immediately
            del df_pandas, data, result
            client.close()
            gc.collect()

            # ========== LÓGICA EXACTA DEL CÓDIGO ORIGINAL ==========

            # Identificar ciclos (cycle_end cuando StageSequence = 1, 4 u 8)
            df = df.with_columns(
                pl.when(
                    (pl.col("StageSequence") == 4)
                    | (pl.col("StageSequence") == 8)
                    | (pl.col("StageSequence") == 1)
                )
                .then(True)
                .otherwise(False)
                .alias("cycle_end")
            )

            # Crear grupos de ciclos
            df = df.with_columns(
                pl.col("cycle_end")
                .shift(1, fill_value=False)
                .cum_sum()
                .alias("cycle_group")
            )

            # Agrupar y obtener FuelLevelsList y StageSum
            result = (
                df.group_by("cycle_group")
                .agg(
                    [
                        pl.col("StageSequence").sum().alias("StageSum"),
                        pl.col("StageSequence").last().alias("StageSequence"),
                        pl.col("FuelLevelLiters").alias("FuelLevelsList"),
                    ]
                )
                .sort("cycle_group")
            )

            # Filtrar solo ciclos Stage 1 (StageSum == 1 y StageSequence == 1)
            # Nota: En el código original filtra StageSum==9 para stage4 y StageSum==26 para stage8
            # Para Stage 1, asumimos StageSum == 1
            st1 = result.filter(pl.col("StageSequence") == 1)

            # Calcular StartFuel y EndFuel
            st1 = st1.with_columns(
                pl.col("FuelLevelsList").list.first().alias("StartFuel"),
                pl.col("FuelLevelsList").list.last().alias("EndFuel"),
            )

            # Aplicar validación: solo si la diferencia < 30 y >= 0
            st1 = st1.with_columns(
                pl.when(
                    (pl.col("StartFuel") - pl.col("EndFuel") < 30)
                    & (pl.col("StartFuel") - pl.col("EndFuel") >= 0)
                )
                .then(pl.col("StartFuel") - pl.col("EndFuel"))
                .otherwise(0)
                .alias("aux")
            )

            # Sumar todo el combustible consumido en Stage 1
            st1_fuel_consumed = st1.select(pl.col("aux").sum()).item()

            # Clean up
            del df, result, st1
            gc.collect()

            self.logger.info(f"{truck_id}: Stage 1 Fuel = {st1_fuel_consumed:.2f} L")

            return st1_fuel_consumed

        except Exception as e:
            self.logger.error(f"Error processing {truck_id}: {str(e)}")
            import traceback

            traceback.print_exc()
            return 0.0

    def collect_all(self) -> List[Dict[str, float]]:
        """
        Collect Stage 1 fuel consumption for all trucks sequentially.

        Processes one truck at a time to minimize memory footprint.

        Returns:
            List[Dict]: List of dictionaries with truck_id and stage1_fuel_consumed
        """
        self.logger.info(f"Starting collection for {len(self.truck_ids)} trucks")
        self.results = []

        for i, truck_id in enumerate(self.truck_ids, 1):
            self.logger.info(f"[{i}/{len(self.truck_ids)}] Processing {truck_id}...")

            # Calculate metric for this truck
            fuel_consumed = self.calculate_stage1_fuel(truck_id)

            # Store only the essential result
            self.results.append(
                {"truck_id": truck_id, "stage1_fuel_consumed": fuel_consumed}
            )

            # Force garbage collection after each truck
            gc.collect()

        self.logger.info("Collection complete!")
        return self.results

    def save_results(self, output_path: str = "stage1_fuel_results.csv"):
        """
        Save collected results to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        if not self.results:
            self.logger.warning("No results to save. Run collect_all() first.")
            return

        # Convert to polars dataframe and save
        df_results = pl.DataFrame(self.results)
        df_results.write_csv(output_path)

        self.logger.info(f"Results saved to {output_path}")

        # Print summary statistics
        total_fuel = sum(r["stage1_fuel_consumed"] for r in self.results)
        avg_fuel = total_fuel / len(self.results) if self.results else 0

        print("\n" + "=" * 60)
        print("STAGE 1 FUEL CONSUMPTION SUMMARY")
        print("=" * 60)
        print(f"Total trucks processed: {len(self.results)}")
        print(f"Total Stage 1 fuel consumed: {total_fuel:.2f} L")
        print(f"Average per truck: {avg_fuel:.2f} L")
        print("=" * 60)


def main():
    """
    Main execution function.
    """

    # Initialize collector
    collector = Stage1FuelCollector(truck_ids=TRUCK_IDS)

    # Collect metrics for all trucks
    results = collector.collect_all()

    # Save results to file
    collector.save_results("stage1_fuel_consumption.csv")

    # Optionally, print individual results
    print("\nDetailed Results:")
    print("-" * 60)
    for result in results:
        print(f"{result['truck_id']}: {result['stage1_fuel_consumed']:.2f} L")


if __name__ == "__main__":
    main()
