from pathlib import Path
import polars as pl
from etl_core.extract import CSVExtractor


def main() -> None:
    truck_id = "T-210"
    dataset = "train_data"
    print(f"🧪 Probando extractor para camión {dataset, truck_id}")

    try:
        extractor: CSVExtractor = CSVExtractor(dataset, truck_id)
        datasets: dict[str, pl.DataFrame] = extractor.load_data()
        unsupported_files: list[str] = extractor.unsupported_files
        print("\n✅ Resultados:")
        for data_type, df in datasets.items():
            print(f"\n📊 {data_type.upper()}:")
            print(f"   - Registros: {df.height}")
            print(f"   - Columnas: {len(df.columns)}")
            if not df.is_empty():
                print(f"   - Muestra:\n{df.head(2)}")

        if unsupported_files:
            print("\n⚠️ Archivos no soportados:")
            for file in unsupported_files:
                print(f"   - {Path(file).name}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
