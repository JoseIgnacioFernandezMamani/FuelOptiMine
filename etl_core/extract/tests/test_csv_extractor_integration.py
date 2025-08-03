import polars as pl
from pathlib import Path
from etl_core.extract import CSVExtractor


# Test: Ejecución completa de load_data() con archivo CSV válido
def test_load_data_integration(tmp_path, monkeypatch):
    # Estructura esperada: /data-set/train_data_sensor/T-001/T-001_sensor.csv
    base_dir = tmp_path / "data-set"
    data_dir = base_dir / "train_data_sensor" / "T-001"
    data_dir.mkdir(parents=True)
    file = data_dir / "T-001_sensor.csv"
    file.write_text("x,y,z\n1,2,3")

    # Mock en settings
    monkeypatch.setattr("etl_core.extract.config.settings.DATA_DIR", base_dir)

    # Mock columnas esperadas
    from etl_core.extract.models.schemas import COLUMN_MAPPING

    COLUMN_MAPPING["sensor"] = ["x", "y", "z"]

    extractor = CSVExtractor(dataset="train_data", truck="T-001")
    result, unsupported = extractor.load_data()

    assert "sensor" in result
    assert isinstance(result["sensor"], pl.DataFrame)
    assert result["sensor"].shape[0] >= 1
    assert all(isinstance(f, str) for f in unsupported)
