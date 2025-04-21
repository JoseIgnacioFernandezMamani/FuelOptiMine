# etl/pipeline/base_pipeline.py
from abc import ABC, abstractmethod

class Pipeline(ABC):
    """Interfaz para todos los pipelines ETL."""
    
    @abstractmethod
    def extract(self):
        """Extrae datos de la fuente."""
        pass
    
    @abstractmethod
    def transform(self, data):
        """Transforma los datos extraídos."""
        pass
    
    @abstractmethod
    def load(self, data):
        """Carga los datos transformados."""
        pass
    
    def run(self):
        """Ejecuta el pipeline completo."""
        data = self.extract()
        transformed_data = self.transform(data)
        return self.load(transformed_data)