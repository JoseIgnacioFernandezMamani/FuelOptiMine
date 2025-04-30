from pathlib import Path
from typing import Union, List, Tuple
from glob import glob
from extract.models.schemas import SUPPORTED_FORMATS, DATASET_TYPES

def get_file_extension(file_path: Union[str, Path], format: str) -> str:
    """Validates and returns file extension for specified format."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension not in SUPPORTED_FORMATS[format]:
        supported = ", ".join(SUPPORTED_FORMATS[format])
        raise ValueError(f"Unsupported extension '{extension}' for {format}. Valid: {supported}")
    return extension

def validate_extension(file_path: Union[str, Path], format: str) -> bool:
    """Silently validates file extension."""
    try:
        get_file_extension(file_path, format)
        return True
    except ValueError:
        return False

def filter_supported_files(files: List[str], format: str) -> Tuple[List[str], List[str]]:
    """Classifies files by format support."""
    return (
        [f for f in files if validate_extension(f, format)],
        [f for f in files if not validate_extension(f, format)]
    )

def validate_truck_exists(base_dir: Path, dataset: str, truck: str, file_extension: str) -> bool:
    """Validates truck data existence in specified dataset."""
    if dataset not in DATASET_TYPES:
        raise ValueError(f"Invalid dataset. Valid options: {DATASET_TYPES}")
    
    patterns = generate_file_patterns(
        base_dir=base_dir,
        dataset=dataset,
        truck=truck,
        data_type="*",  # All data types
        file_extension=file_extension
    )
    
    if not find_matching_files(patterns):
        raise FileNotFoundError(f"No files found for truck {truck} in {dataset} dataset")
    return True

def generate_file_patterns(base_dir: Path, dataset: str, truck: str, 
                          data_type: str, file_extension: str) -> List[str]:
    """Generates search patterns for truck data files."""
    if dataset not in DATASET_TYPES:
        raise ValueError(f"Invalid dataset type. Valid: {DATASET_TYPES}")
    
    return [
        str(base_dir / f"{dataset}_{data_type}" / "*" / f"{truck}_*.{file_extension.lstrip('.')}")
    ]

def find_matching_files(patterns: List[str]) -> List[str]:
    """Finds files matching glob patterns."""
    return [file for pattern in patterns for file in glob(pattern, recursive=True)]