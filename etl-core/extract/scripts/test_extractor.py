from pathlib import Path

from extract.implementations.local.csv_extractor import CSVExtractor
from extract.config.settings import DATA_DIR

def main():
        truck_id = "T-210"
        dataset = "train_data"
        print(f"🧪 Probando extractor para camión {dataset, truck_id}")
        
        try:
            extractor = CSVExtractor(dataset, truck_id)
            datasets, unsupported_files = extractor.load_data()  
            
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