from typing import Optional
from .core import CriticalExtractionError

class InvalidDatasetError(CriticalExtractionError):
    """Invalid dataset"""
    def __init__(self, dataset: str, valid_options: list, cause: Optional[Exception] = None):
        super().__init__(
            message=f"Invalid dataset. Allowed options: {valid_options}",
            cause=cause,
            dataset=dataset
        )

class SchemaValidationError(CriticalExtractionError):
    """Data structure error"""
    def __init__(self, data_type: str, missing_columns: list, cause: Optional[Exception] = None, **kwargs):
        super().__init__(
            message=f"Invalid schema for {data_type}. Missing columns: {missing_columns}",
            cause=cause,
            **kwargs
        )