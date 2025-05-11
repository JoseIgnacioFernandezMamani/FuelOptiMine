from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import polars as pl

class IBaseExtractor(ABC):
    """Base interface for all data extractors"""
    
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
