# tests/etl/test_extractors.py
import unittest
import pandas as pd
import tempfile
import os

from src.etl.extract.local.csv_extractor import CSVExtractor

class TestCSVExtractor(unittest.TestCase):
    def setUp(self):
        # Crear un archivo CSV temporal para las pruebas
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test.csv")
        
        # Crear un DataFrame de prueba y guardarlo como CSV
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        self.test_df.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        # Limpiar archivos temporales
        os.remove(self.csv_path)
        os.rmdir(self.temp_dir)
    
    def test_extract(self):
        # Probar el extractor CSV
        extractor = CSVExtractor(self.csv_path)
        result = extractor.extract()
        
        # Verificar que el resultado es igual al DataFrame original
        pd.testing.assert_frame_equal(result, self.test_df)
    
    def test_invalid_path(self):
        # Probar que se lanza una excepción con una ruta inválida
        with self.assertRaises(FileNotFoundError):
            extractor = CSVExtractor("ruta/inexistente.csv")