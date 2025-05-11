import polars as pl
from pathlib import Path
from typing import Any, Dict
from analitycs.EDA.config.settings import DATA_DIR


class FuelAnalysisOptimized:
    def __init__(self) -> None:
        self.data_path = Path(DATA_DIR) / "T-210_sensor.csv"
        self.sensor_df: pl.DataFrame = None
        self.min_date = None
        self.max_date = None

    def load_sensor_data(self) -> pl.DataFrame:
        """Load and preprocess sensor data with proper type casting"""
        try:
            self.sensor_df = (
                pl.read_csv(
                    self.data_path,
                    columns=[
                        "TimeStamp",
                        "RecordDuration",
                        "FuelLevelLiters",
                        "Speed",
                        "RPM",
                    ],
                    try_parse_dates=True,
                )
                .sort("TimeStamp")
                .with_columns(
                    pl.col("FuelLevelLiters").diff().alias("delta_fuel"),
                )
            )

            # Extract temporal range
            self.min_date = self.sensor_df["TimeStamp"].min()
            self.max_date = self.sensor_df["TimeStamp"].max()

            return self.sensor_df

        except FileNotFoundError:
            raise RuntimeError(f"Data file not found: {self.data_path}")

    def generate_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Generate comprehensive statistics for numerical and temporal columns"""
        if self.sensor_df is None:
            raise RuntimeError("Data not loaded. Call load_sensor_data() first")

        stats = {}

        # Select relevant columns and cast types
        df = self.sensor_df

        for col in df.columns:
            col_type = df.schema[col]
            col_stats = {}

            if col == "TimeStamp":  # Handle datetime separately
                col_stats = {
                    "first_record": df[col].min(),
                    "last_record": df[col].max(),
                    "total_duration": (df[col].max() - df[col].min()).total_seconds(),
                }
            elif col_type in [pl.Float64, pl.Int64]:  # Numerical columns
                # Calculate quantiles correctly
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)

                col_stats = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "mode": (
                        df[col].mode().item() if not df[col].mode().is_empty() else None
                    ),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "variance": df[col].var(),
                    "std_dev": df[col].std(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1 if q1 is not None and q3 is not None else None,
                    "skewness": df[col].skew(),
                    "kurtosis": df[col].kurtosis(),
                    "non_null_count": df[col].count(),
                    "null_count": df[col].null_count(),
                }

            stats[col] = col_stats

        return stats


if __name__ == "__main__":
    try:
        analyzer = FuelAnalysisOptimized()
        df = analyzer.load_sensor_data()
        print(f"Data loaded successfully. Records: {len(df):,}")

        print(df.head(5))

        stats = analyzer.generate_statistics()
        print("\nFuel Level Statistics:")
        fuel_stats = stats["FuelLevelLiters"]
        print(f"Average: {fuel_stats['mean']:.2f} L")
        print(f"Range: {fuel_stats['min']:.2f}-{fuel_stats['max']:.2f} L")
        print(f"Data Span: {stats['TimeStamp']['total_duration']/3600:.2f} hours")

    except Exception as e:
        print(f"Error: {str(e)}")
