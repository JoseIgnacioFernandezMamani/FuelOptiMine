import unittest
from unittest.mock import patch, MagicMock
from etl_core.load.implementations.sensor.sensor_loader import SensorLoader
from etl_core.utils.sensor_schemas import SensorSchema
import polars as pl


class TestSensorLoader(unittest.TestCase):

    @patch("etl.load.base_loader.connect")
    def test_load_data(self, mock_connect):
        # Configurar mocks
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value.cursor.return_value = (
            mock_cursor
        )

        # Crear datos de prueba
        test_data = [
            {
                "ShiftDate": "2023-01-01",
                "TimeStamp": "2023-01-01 12:00:00",
                "Latitude": 3600000,  # 1 grado
                "Longitude": -7200000,  # -2 grados
                "Elevation": 0,
                "Equipment": "EX001",
            }
        ]

        # Ejecutar carga
        loader = SensorLoader()
        result = loader.load_data(test_data, batch_size=1000)

        # Verificar llamadas
        self.assertEqual(result, 1)
        mock_cursor.executemany.assert_called_once()

        # Verificar transformación de coordenadas
        inserted_data = mock_cursor.executemany.call_args[0][1]
        self.assertAlmostEqual(inserted_data[0][2], 1.0)  # Latitude
        self.assertAlmostEqual(inserted_data[0][3], -2.0)  # Longitude

    def test_coordinate_conversion(self):
        loader = SensorLoader()
        record = {"Latitude": 3600000, "Longitude": -1800000, "Elevation": 900000}
        transformed = loader._transform_record(record)

        self.assertAlmostEqual(transformed["Latitude"], 1.0)
        self.assertAlmostEqual(transformed["Longitude"], -0.5)
        self.assertAlmostEqual(transformed["Elevation"], 0.25)
