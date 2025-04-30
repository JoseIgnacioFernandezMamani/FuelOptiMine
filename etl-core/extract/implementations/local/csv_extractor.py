from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import polars as pl
import pandas as pd
from pathlib import Path

from extract.interfaces.base import IBaseExtractor
from extract.utils.file_utils import (
    validate_truck_exists,
    generate_file_patterns,
    find_matching_files,
    filter_supported_files,
    get_file_extension
)
from extract.models.schemas import COLUMN_MAPPING, DATASET_TYPES
from extract.config.settings import DATA_DIR

class CSVExtractor(IBaseExtractor):
    """Extracts tabular data from multiple file formats for specific trucks"""
    
    def __init__(self, dataset: str, truck: str):
        if dataset not in DATASET_TYPES:
            raise ValueError(f"Invalid dataset. Valid options: {DATASET_TYPES}")
        
        self.dataset = dataset
        self.truck = truck.upper()
        self.base_dir = Path(DATA_DIR)
        self.unsupported_files = []
        self.FORMAT = "tabular"

    @staticmethod
    def _detect_separator(file_path: str) -> str:
        """Detects delimiter for CSV/TSV files."""
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            line = f.readline()
            return next((sep for sep in [';', '\t', ','] if sep in line), ',')

    def _load_single_file(self, file_path: str, data_type: str) -> pl.DataFrame:
        """Loads individual data file based on its format."""
        try:
            ext = get_file_extension(file_path, self.FORMAT)
            
            # Handle delimited files
            if ext in ('.csv', '.tsv'):
                return pl.read_csv(
                    file_path,
                    skip_rows=1,
                    separator=self._detect_separator(file_path),
                    has_header=False,
                    new_columns=COLUMN_MAPPING[data_type],
                    dtypes={col: pl.String for col in COLUMN_MAPPING[data_type]},
                    encoding='utf8',
                    ignore_errors=True
                )
                
            # Handle binary formats
            elif ext in ('.feather'):
                return pl.read_parquet(file_path) 
            
            # Handle parquet files
            elif ext in ('.parquet'):
                return pl.read_ipc(file_path)
                
            # Handle Excel files
            elif ext in ('.xls', '.xlsx'):
                df = pd.read_excel(
                    file_path, 
                    skiprows=1, 
                    header=None, 
                    engine='openpyxl'
                )
                return pl.from_pandas(df).rename({i: col for i, col in enumerate(COLUMN_MAPPING[data_type])})
                
            else:
                raise ValueError(f"Unsupported file format: {ext}")

        except Exception as e:
            print(f"Error loading {Path(file_path).name}: {str(e)}")
            self.unsupported_files.append(file_path)
            return pl.DataFrame()

    def load_data(self) -> Tuple[Dict[str, pl.DataFrame], List[str]]:
        """Main method to load and consolidate truck data."""
        validate_truck_exists(
            base_dir=self.base_dir,
            dataset=self.dataset,
            truck=self.truck,
            file_extension="*"
        )
        
        datasets = {}
        
        for data_type in COLUMN_MAPPING:
            try:
                # Generate search patterns
                patterns = generate_file_patterns(
                    base_dir=self.base_dir,
                    dataset=self.dataset,
                    truck=self.truck,
                    data_type=data_type,
                    file_extension="*"
                )
                
                # Process files
                all_files = find_matching_files(patterns)
                valid_files, invalid_files = filter_supported_files(all_files, self.FORMAT)
                self.unsupported_files.extend(invalid_files)
                
                if not valid_files:
                    print(f"No valid files found for {data_type}")
                    continue
                
                # Parallel processing
                with ThreadPoolExecutor() as executor:
                    dfs = list(executor.map(
                        lambda f: self._load_single_file(f, data_type),
                        valid_files
                    ))
                
                # Combine results
                combined_df = pl.concat([df for df in dfs if not df.is_empty()])
                if "TimeStamp" in combined_df.columns:
                    combined_df = combined_df.sort("TimeStamp")
                
                datasets[data_type] = combined_df
                print(f"[{data_type.upper()}] Loaded records: {combined_df.height}")

            except KeyError:
                print(f"Missing schema definition for {data_type}")
                datasets[data_type] = pl.DataFrame()
        
        return datasets, self.unsupported_files