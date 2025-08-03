import pytest
from pathlib import Path
from etl_core.extract import CSVExtractor
from etl_core.extract.exceptions import (
    InvalidDatasetError,
    UnsupportedFormatError,
    SchemaValidationError,
)


# Test: Dataset inválido
def test_invalid_dataset_error():
    with pytest.raises(InvalidDatasetError):
        CSVExtractor(dataset="invalid_data", truck="T-001")


# Test: Formato no soportado
def test_unsupported_format_error(tmp_path):
    file = tmp_path / "archivo.unsupported"
    file.write_text("contenido irrelevante")

    extractor = CSVExtractor(dataset="train_data", truck="T-001")
    from etl_core.extract.models.schemas import COLUMN_MAPPING

    COLUMN_MAPPING["sensor"] = ["col1"]

    with pytest.raises(UnsupportedFormatError):
        extractor._load_single_file(str(file), "sensor")


# Test: Columnas inválidas en Excel (menos columnas que el esquema)
def test_schema_validation_error(tmp_path):
    import pandas as pd

    file = tmp_path / "bad_excel.xlsx"
    df = pd.DataFrame([[1, 2]], columns=["a", "b"])
    df.to_excel(file, index=False, header=None, engine="openpyxl")

    extractor = CSVExtractor(dataset="train_data", truck="T-001")
    from etl_core.extract.models.schemas import COLUMN_MAPPING

    COLUMN_MAPPING["sensor"] = ["col1", "col2", "col3"]  # Se esperan 3 columnas

    with pytest.raises(SchemaValidationError):
        extractor._load_single_file(str(file), "sensor")
