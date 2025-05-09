import polars as pl
from abc import abstractmethod

from etl_core.extract.interfaces.base import IBaseExtractor

class IFileExtractor(IBaseExtractor):
    """Interface for file-based data extractors"""
    
    @abstractmethod
    def _load_single_file(self, file_path: str, data_type: str) -> pl.DataFrame:
        """Load individual file implementation"""
        pass
