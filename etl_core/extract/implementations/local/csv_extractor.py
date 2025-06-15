from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import polars as pl
import pandas as pd
from pathlib import Path
import csv

from etl_core.extract.interfaces.local import IFileExtractor
from etl_core.extract.utils import (
    get_file_extension,
    filter_supported_files,
    validate_truck_exists,
    generate_file_patterns,
    find_matching_files,
)
from etl_core.extract.exceptions import (
    DataLoadingWarning,
    InvalidDatasetError,
    SchemaValidationError,
    UnsupportedFormatError,
    RecoverableExtractionError,
    CriticalExtractionError,
)
from etl_core.utils import TRUCK_SPECS
from etl_core.extract.models.schemas import (
    COLUMN_MAPPING,
    DATASET_TYPES,
    SUPPORTED_FORMATS,
    FUEL_SUPPLY,
)
from etl_core.extract.config.settings import DATA_DIR


class CSVExtractor(IFileExtractor):
    """Extracts and consolidates tabular data from multiple file formats for specific trucks"""

    def __init__(self, dataset: str, truck: str):
        """Initialize extractor with validation

        Args:
            dataset: Name of dataset (train_data, val_data, test_data)
            truck: Truck identifier (format T-XXX)

        Raises:
            InvalidDatasetError: For unsupported dataset types
        """
        if dataset not in DATASET_TYPES:
            raise InvalidDatasetError(dataset, DATASET_TYPES)

        self.dataset = dataset
        self.truck = truck.upper().strip()
        self.base_dir = Path(DATA_DIR)
        self.unsupported_files = []
        self.FORMAT = "tabular"
        self.truck_number = self.truck.replace("T-", "").strip()
        self.VALID_TRUCK_MODELS = {"CAT - 789C", "CAT - 793D"}

    @staticmethod
    def _detect_separator_and_header(file_path: str) -> str:
        """Detects CSV/TSV file delimiter from first line

        Args:
            file_path: Path to file

        Returns:
            Detected separator (;, \t, or ,)
        """
        with open(file_path, "r", encoding="utf-8-sig") as f:
            line1 = f.readline()
            line2 = f.readline()
            sample = line1 + line2
            f.seek(0)
            first_line = f.readline()

        dialect = csv.Sniffer().sniff(first_line)
        separator = dialect.delimiter

        header = csv.Sniffer().has_header(sample)

        return separator, header

    def _load_single_file(self, file_path: str, data_type: str) -> pl.DataFrame:
        """Load individual data file with format-specific handling

        Args:
            file_path: Path to data file
            data_type: Type of data (sensor, cycle, time_model, dispatch)

        Returns:
            Parsed DataFrame

        Raises:
            UnsupportedFormatError: For unrecognized file formats
            DataLoadingWarning: For recoverable parsing errors
        """
        try:
            ext = get_file_extension(file_path, self.FORMAT)
            columns = COLUMN_MAPPING[data_type]

            # CSV/TSV handling
            if ext in (".csv", ".tsv"):

                separator, header_file = self._detect_separator_and_header(file_path)

                df = pl.read_csv(
                    file_path,
                    skip_rows=1 if header_file else 0,
                    separator=separator,
                    has_header=False,
                    new_columns=columns,
                    dtypes={col: pl.String for col in columns},
                    encoding="utf8",
                    ignore_errors=True,
                )

            # Feather format
            elif ext == ".feather":
                df = pl.read_ipc(file_path)

            # Parquet format
            elif ext == ".parquet":
                df = pl.read_parquet(file_path)

            # Excel handling
            elif ext in (".xls", ".xlsx"):
                pandas_df = pd.read_excel(
                    file_path, skiprows=1, header=None, engine="openpyxl"
                )
                # Validate column count
                if pandas_df.shape[1] != len(COLUMN_MAPPING[data_type]):
                    raise SchemaValidationError(
                        data_type=data_type,
                        message=f"Expected {len(COLUMN_MAPPING[data_type])} columns, found {pandas_df.shape[1]}",
                    )
                df = pl.from_pandas(pandas_df).rename(
                    {i: col for i, col in enumerate(COLUMN_MAPPING[data_type])}
                )

            else:
                raise UnsupportedFormatError(
                    file_path=file_path,
                    format=self.FORMAT,
                    supported_formats=SUPPORTED_FORMATS[self.FORMAT],
                )

            return df

        except Exception as e:
            self.unsupported_files.append(file_path)
            if isinstance(e, (UnsupportedFormatError, SchemaValidationError)):
                raise
            raise DataLoadingWarning(
                file_path=file_path, details=str(e), dataset=self.dataset, cause=e
            ) from e

    def _load_fuel_supply(self) -> pl.DataFrame:
        """Load and transform fuel supply data from Excel files

        Returns:
            pl.DataFrame: Consolidated DataFrame with transformed fuel supply data
        """
        try:
            # Generate file search pattern
            pattern = generate_file_patterns(
                base_dir=self.base_dir,
                dataset=self.dataset,
                data_type="fuel_supply",
                truck="*",
                file_extension="*",
            )

            # Find matching files
            supply_files = find_matching_files(pattern)
            if not supply_files:
                return pl.DataFrame()

            dfs = []
            # Get truck capacity from specs
            truck_capacity = TRUCK_SPECS.get(self.truck, {}).get("capacity", 3000)

            for file in supply_files:
                try:
                    file_name = Path(file).stem
                    parts = file_name.split("_")
                    origin = parts[1] if len(parts) > 1 else "Unknown"

                    # Read Excel file
                    pandas_df = pd.read_excel(
                        file,
                        header=0,
                        engine="openpyxl",
                        usecols=FUEL_SUPPLY,
                    )

                    # Convert to Polars DataFrame
                    df = pl.from_pandas(pandas_df)

                    # Handle timestamp column conversion
                    time_col = "fin_desp"
                    if time_col in df.columns:
                        if df[time_col].dtype in (
                            pl.Datetime,
                            pl.Datetime("ms"),
                            pl.Datetime("us"),
                            pl.Datetime("ns"),
                        ):
                            df = df.rename({time_col: "TimeStamp"})
                        else:
                            df = df.with_columns(
                                pl.col(time_col).cast(pl.Utf8),
                                pl.col(time_col)
                                .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
                                .alias("TimeStamp"),
                            )
                    else:
                        # Create empty timestamp column if missing
                        df = df.with_columns(
                            pl.lit(None).alias("TimeStamp").cast(pl.Datetime)
                        )

                    # Apply transformations
                    df = df.with_columns(
                        # Add origin from file name
                        pl.lit(origin).alias("Origin").cast(pl.Utf8),
                        # Extract equipment number (first 3 digits)
                        pl.col("Veh")
                        .str.slice(0, 3)
                        .map_elements(
                            lambda x: f"T-{x}" if x else None, return_dtype=pl.Utf8
                        )
                        .alias("Equipment"),
                        # Normalize truck model name
                        pl.col("Descripcion")
                        .str.replace_all(r"[\s-]+", "", literal=True)
                        .alias("TruckFleet"),
                        # Convert fuel volume to float
                        pl.col("volumCorregido")
                        .cast(pl.Float64, strict=False)
                        .alias("FuelLevelLiters"),
                    )

                    # Calculate derived fields
                    df = df.with_columns(
                        # Extract date from timestamp
                        pl.col("TimeStamp").dt.date().alias("ShiftDate"),
                        # Determine shift (Day/Night)
                        pl.when(
                            (pl.col("TimeStamp").dt.hour() >= 7)
                            & (pl.col("TimeStamp").dt.hour() <= 19)
                        )
                        .then(pl.lit("D"))
                        .otherwise(pl.lit("N"))
                        .alias("Shift"),
                        # Calculate fuel percentage
                        (pl.col("FuelLevelLiters") / truck_capacity * 100)
                        .clip(upper_bound=100.0)
                        .round(4)
                        .alias("FuelLevel"),
                    )

                    # Filter records
                    df = df.filter(
                        (pl.col("TruckFleet").is_in(self.VALID_TRUCK_MODELS))
                        & (pl.col("Equipment") == self.truck)
                    )

                    # Select final columns
                    final_cols = [
                        "Origin",
                        "ShiftDate",
                        "TimeStamp",
                        "Equipment",
                        "TruckFleet",
                        "FuelLevelLiters",
                        "Shift",
                        "FuelLevel",
                    ]
                    df = df.select([col for col in final_cols if col in df.columns])

                    dfs.append(df)

                except Exception as e:
                    # Track unsupported files and log error
                    self.unsupported_files.append(file)
                    print(f"⚠️ Error processing {Path(file).name}: {str(e)}")

            # Return concatenated results
            return pl.concat(dfs) if dfs else pl.DataFrame()

        except Exception as e:
            self.unsupported_files.append(file)
            raise DataLoadingWarning(
                file_path=file, details=str(e), dataset=self.dataset, cause=e
            ) from e

    def load_data(self) -> Tuple[Dict[str, pl.DataFrame], List[str]]:
        """Main data loading pipeline

        Returns:
            Tuple containing:
                - Dictionary of DataFrames by data type
                - List of unsupported files

        Raises:
            CriticalExtractionError: For unrecoverable failures
        """
        try:
            # Validate truck existence
            validate_truck_exists(
                base_dir=self.base_dir,
                dataset=self.dataset,
                truck=self.truck,
                file_extension="*",
            )

            datasets = {}

            # Process core data types
            for data_type in COLUMN_MAPPING:
                try:
                    # Generate search patterns
                    patterns = generate_file_patterns(
                        base_dir=self.base_dir,
                        dataset=self.dataset,
                        data_type=data_type,
                        truck=self.truck,
                        file_extension="*",
                    )

                    # Locate and filter files
                    all_files = find_matching_files(patterns)
                    valid_files, invalid_files = filter_supported_files(
                        all_files, self.FORMAT
                    )
                    self.unsupported_files.extend(invalid_files)

                    if not valid_files:
                        print(f"[Warning] No valid files found for {data_type}")
                        continue

                    # Parallel file processing
                    dfs = []
                    with ThreadPoolExecutor() as executor:
                        futures = {
                            executor.submit(self._load_single_file, f, data_type): f
                            for f in valid_files
                        }

                        for future in as_completed(futures):
                            file_path = futures[future]
                            try:
                                df = future.result()
                                if not df.is_empty():
                                    dfs.append(df)
                            except RecoverableExtractionError as e:
                                print(f"[Recoverable] {str(e)}")

                    # Skip if no valid data
                    if not dfs:
                        print(f"[Warning] Empty data after processing {data_type}")
                        continue

                    # Consolidate and validate
                    combined_df = pl.concat(dfs)
                    if "TimeStamp" in combined_df.columns:
                        combined_df = combined_df.sort("TimeStamp")

                    # Schema validation
                    missing_cols = [
                        col
                        for col in COLUMN_MAPPING[data_type]
                        if col not in combined_df.columns
                    ]
                    if missing_cols:
                        raise SchemaValidationError(
                            data_type=data_type, missing_columns=missing_cols
                        )

                    datasets[data_type] = combined_df

                except RecoverableExtractionError as e:
                    print(f"[Recoverable] {e}")
                except CriticalExtractionError as e:
                    print(f"[Critical] {e}")
                    raise

            # Load dispatch data
            try:
                dispatch_df = self._load_fuel_supply()
                if not dispatch_df.is_empty():
                    datasets["fuel_supply"] = dispatch_df
            except RecoverableExtractionError as e:
                print(f"[Recoverable] Dispatch data error: {e}")

            return datasets, self.unsupported_files

        except CriticalExtractionError as e:
            print(f"[Critical] Pipeline failed: {e}")
            raise
