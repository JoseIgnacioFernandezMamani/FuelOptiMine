from etl_core.load.implementations.core.base_loader import BaseClickHouseLoader
from etl_core.load.implementations.sensor.sensor_loader import SensorSchema


class SensorLoader(BaseClickHouseLoader):
    """Loader especializado para datos de sensores"""

    def __init__(self, **kwargs):
        super().__init__(schema=SensorSchema, table_name="sensor_data", **kwargs)

    def _milliarcseconds_to_degrees(self, value):
        """Convierte milliarcseconds a grados decimales"""
        return value / 3600000.0

    def _transform_record(self, record):
        """Aplica transformaciones específicas de sensores"""
        # Convertir coordenadas
        record["Latitude"] = self._milliarcseconds_to_degrees(record["Latitude"])
        record["Longitude"] = self._milliarcseconds_to_degrees(record["Longitude"])
        record["Elevation"] = self._milliarcseconds_to_degrees(record["Elevation"])
        return record
