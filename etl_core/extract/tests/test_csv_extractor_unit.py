import polars as pl
from pathlib import Path
from etl_core.extract import CSVExtractor, COLUMN_MAPPING

# mock data for testing
CSV_CONTENT = (
    "ShiftDate;Shift;TimeStamp;RecordDuration;Equipment;TruckFleet;FuelLevel;FuelLevelLiters;FuelGauge;Speed;RPM;Ralenti;Latitude;Longitude;Elevation\n"
    "2024-02-01;D;2024-02-01 07:01:30.000;NULL;T-210;CAT 789C;31.74;1015.68000000;Medium;0;0;Ralenti;-76001573;-241968218;417059\n"
    "2024-02-01;D;2024-02-01 07:04:30.000;180;T-210;CAT 789C;31.07;994.24000000;Medium;22;0;Moviendose;-75996785;-241969167;416300\n"
    "2024-02-01;D;2024-02-01 07:05:30.000;60;T-210;CAT 789C;30.985;991.68000000;Medium;32;0;Moviendose;-75999602;-241955269;411169"
)

# mock schema for testing
SCHEMA_SENSOR: list[str] = [
    "ShiftDate",
    "Shift",
    "TimeStamp",
    "RecordDuration",
    "Equipment",
    "TruckFleet",
    "FuelLevel",
    "FuelLevelLiters",
    "FuelGauge",
    "Speed",
    "RPM",
    "Ralenti",
    "Latitude",
    "Longitude",
    "Elevation",
]


# Test: _detect_separator_and_header()
def test_detect_separator_and_header_csv(tmp_path) -> None:
    file: Path = tmp_path / "test.csv"
    file.write_text(CSV_CONTENT)

    sep, header = CSVExtractor._detect_separator_and_header(str(file))
    assert sep == ";"
    assert header is True


# Test: _load_single_file con CSV válido
def test_load_single_file_csv(tmp_path, monkeypatch) -> None:
    # Crear estructura de directorios simulada
    dataset_dir = tmp_path / "train_data" / "T-210"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    file = dataset_dir / "T-210_sensor.csv"
    file.write_text(CSV_CONTENT)

    # Mockear DATA_DIR para que apunte al directorio temporal
    monkeypatch.setattr("etl_core.extract.config.settings.DATA_DIR", str(tmp_path))

    # Backup del mapeo original
    original_mapping = COLUMN_MAPPING.copy()

    try:
        # Mock schema
        COLUMN_MAPPING["sensor"] = SCHEMA_SENSOR

        # Crear extractor con dataset válido
        extractor = CSVExtractor(dataset="train_data", truck="T-210")

        # Cargar el archivo
        df = extractor._load_single_file(str(file), data_type="sensor")

        # Verificaciones básicas
        assert df.shape == (3, 15)
        assert df.columns == SCHEMA_SENSOR

        # Verificar que todas las columnas son strings
        for col in df.columns:
            assert df[col].dtype == pl.Utf8, f"Columna {col} no es String"

        # Verificar valores específicos
        assert df["RecordDuration"][0] == "NULL"
        assert df["FuelLevel"][0] == "31.74"
        assert df["Latitude"][0] == "-76001573"

    finally:
        # Restaurar mapeo original
        COLUMN_MAPPING.clear()
        COLUMN_MAPPING.update(original_mapping)


# Test _load_fuel_supply
def test_load_fuel_supply(tmp_path) -> None:
    file: Path = tmp_path / "T-210_fuel_supply.csv"
    file.write_text(
        "Date;Time;Truck;FuelType;Volume\n2024-02-01;07:00;T-210;Diesel;1000"
    )

    # Mock schema
    COLUMN_MAPPING["fuel_supply"] = ["Date", "Time", "Truck", "FuelType", "Volume"]

    extractor: CSVExtractor = CSVExtractor(dataset=str(tmp_path), truck="T-210")
    df: pl.DataFrame = extractor._load_single_file(str(file), data_type="fuel_supply")

    assert df.shape == (1, 5)
    assert df.columns == COLUMN_MAPPING["fuel_supply"]
    assert df["Truck"][0] == "T-210"
