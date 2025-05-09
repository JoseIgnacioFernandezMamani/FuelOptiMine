import unittest
from extract.implementations.local.csv_extractor import CSVExtractor

class TestCSVExtractorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configurar con datos reales (asegurarse de tener datos de prueba)
        cls.valid_truck = "T01"
        cls.test_data_types = ["sensor", "time_model", "cycle"]

    def test_load_real_data(self):
        extractor = CSVExtractor(self.valid_truck)
        data = extractor.load_data()
        
        self.assertIsInstance(data, dict)
        for data_type in self.test_data_types:
            self.assertIn(data_type, data)
            self.assertFalse(data[data_type].is_empty())

if __name__ == "__main__":
    unittest.main()