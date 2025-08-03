import unittest
from unittest.mock import patch
from extract.implementations.local.data_loader import LocalDataLoader
from extract.config.settings import DATA_DIR

class TestLocalDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.valid_truck = "T01"
        cls.invalid_truck = "INVALID"

    def test_invalid_truck_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            LocalDataLoader(self.invalid_truck)

    @patch('glob.glob')
    def test_data_loading(self, mock_glob):
        # Configurar mock
        mock_glob.return_value = [str(DATA_DIR / "train_data_sensor" / "test_1" / "T01_001.csv")]
        
        loader = LocalDataLoader(self.valid_truck)
        data = loader.load_data()
        
        self.assertIn('sensor', data)
        self.assertFalse(data['sensor'].empty)

if __name__ == '__main__':
    unittest.main()