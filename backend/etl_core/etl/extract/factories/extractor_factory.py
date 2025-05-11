from .interfaces.iextractor import IExtractor
from .models.config import DataSourceConfig
from .implementations.local import CSVExtractor, JSONExtractor
from .implementations.external import APIExtractor, DatabaseExtractor

class ExtractorFactory:
    @staticmethod
    def create(config: DataSourceConfig) -> IExtractor:
        if config.source_type == 'csv':
            return CSVExtractor(config)
        elif config.source_type == 'json':
            return JSONExtractor(config)
        elif config.source_type == 'api':
            return APIExtractor(config)
        elif config.source_type == 'database':
            return DatabaseExtractor(config)
        raise ValueError("Tipo de fuente no soportado")