# En extract/tests/unit/test_csv_extractor.py
from extract.implementations.local import CSVExtractor

def test_csv_extractor():
    extractor = CSVExtractor("TRUCK123")
    data = extractor.load_data()
    # Assertions...