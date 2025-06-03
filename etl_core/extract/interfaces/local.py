import polars as pl
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class IFileExtractor(ABC):
    """Interface for file-based data extractors"""

    @abstractmethod
    def load_data(self) -> Tuple[Dict[str, pl.DataFrame], List[str]]:
        """
        Load data from source

        Returns:
            Tuple:
                - Dict with data type as key and DataFrame as value
                - List of unsupported/unprocessable files/sources
        """
        pass

    @abstractmethod
    def _load_single_file(self, file_path: str, data_type: str) -> pl.DataFrame:
        """Load individual file implementation"""
        pass
