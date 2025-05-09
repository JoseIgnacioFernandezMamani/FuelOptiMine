from abc import ABC, abstractmethod
from typing import Dict
import polars as pl

class IDataCleaner(ABC):
    """Interface para operaciones de limpieza"""
    
    @abstractmethod
    def clean_data(self, raw_data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        pass

class IDataTransformer(ABC):
    """Interface para operaciones de transformación"""
    
    @abstractmethod
    def transform_data(self, cleaned_data: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
        pass