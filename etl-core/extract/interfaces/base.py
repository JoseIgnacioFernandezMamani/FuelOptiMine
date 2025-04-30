from abc import ABC, abstractmethod
from typing import Dict, List
import polars as pl

class IBaseExtractor(ABC):
    """Interface defining the contract for data loading classes"""
    
    @abstractmethod
    def load_data(self) -> Dict[str, pl.DataFrame]:
        """
        Load all types of data into separate DataFrames
        
        Returns:
            Dict[str, pl.DataFrame]: Dictionary with data types as keys and DataFrames as values
        """
        pass

    @abstractmethod
    def _load_single_file(self) -> pl.DataFrame:
        """
        Load a single file into a DataFrame
        
        Returns:
            pl.DataFrame: Loaded DataFrame
        """
        pass
    