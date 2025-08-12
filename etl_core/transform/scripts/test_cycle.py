from etl_core.transform.implementation.cycle.transformer import CycleTransformer
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
import sys
import os
import polars as pl


def run_cycle_etl_pipeline():
    # Configuración de rutas y parámetros
    truck_id = "T-211"
    dataset_name = "train_data"
    data_type = "cycle"

    # Configurar rutas de importación
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    print(f"🚚 Starting ETL for truck {truck_id} - {dataset_name} (Cycle Data)")

    try:
        # 1. Data extraction
        print("\n🔍 Extracting data from CSV...")
        extractor = CSVExtractor(dataset_name, truck_id)
        raw_data = extractor.load_data()

        if not raw_data or data_type not in raw_data:
            raise ValueError(
                f"Data of type '{data_type}' not found or invalid structure"
            )

        # Get Polars DataFrame directly
        df_raw: pl.DataFrame = raw_data[data_type]
        print(f"✅ Raw data loaded: {df_raw.height} records")
        print("Initial schema:", df_raw.schema)

        # 2. Data transformation
        print("\n🔄 Processing data with CycleTransformer...")
        transformer = CycleTransformer()
        df_clean = transformer.run_transform(df_raw)

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("Transformation returned empty data")

        # 3. Verify expected columns
        expected_columns = ["ShiftDate", "Shift", "Equipment", "TruckFleet"]

        missing_columns = [
            col for col in expected_columns if col not in df_clean.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")

        # 4. Updated metrics report
        print("\n📊 Final metrics:")
        metrics = [
            ("Initial records", "initial_records"),
            ("After cleaning", "after_cleaning_records"),
            ("After validation", "after_validation_records"),
            ("After transformation", "after_transform_records"),
            ("Null records removed", "removed_null_records"),
            ("Duplicate records removed", "removed_duplicate_records"),
            ("Invalid schema records", "invalid_schema_records"),
            ("Outliers removed", "outliers_removed"),
            ("Invalid geo records", "invalid_geo_records"),
            ("Categorical empty fixed", "categorical_empty_fixed"),
            ("Negative times fixed", "negative_times_fixed"),
            ("Invalid tonnage fixed", "invalid_tonnage_fixed"),
            ("DateTime parsing errors", "datetime_parsing_errors"),
            ("Clean data percentage", "clean_data_percentage"),
            ("Valid data percentage", "valid_data_percentage"),
            ("Final data percentage", "final_data_percentage"),
        ]

        for name, key in metrics:
            value = transformer.metrics.get(key, "N/A")
            if isinstance(value, float):
                print(f"- {name}: {value:.4f}%")
            else:
                print(f"- {name}: {value}")

        # 5. Show transformed data sample
        print("\n🔍 Transformed data sample (First 5 rows):")
        sample_columns = [
            "Equipment",
            "ShiftDate",
            "Shift",
            "Shovel",
            # "E_TravelingStart",
            # "E_TravelingEnd",
            # "E_WaitingStart",
            # "E_WaitingEnd",
            # "E_SpottingStart",
            # "E_SpottingEnd",
            # "Material",
            # "TotalCycleTime",
            # "MeasuredTonnage",
            # "TotalDistance",
            # "TonnageEfficiency",
            # "AverageSpeed",
        ]
        # Filter only existing columns
        existing_sample_columns = [
            col for col in sample_columns if col in df_clean.columns
        ]
        print(df_clean.select(existing_sample_columns).head(5))

        # 6. Show data quality summary
        print("\n📈 Data Quality Summary:")
        total_records = df_clean.height

        # Check for null values in key columns
        key_columns = ["TotalCycleTime", "MeasuredTonnage", "G_Latitude", "G_Longitude"]
        for col in key_columns:
            if col in df_clean.columns:
                null_count = df_clean.filter(pl.col(col).is_null()).height
                null_percentage = (
                    (null_count / total_records) * 100 if total_records > 0 else 0
                )
                print(f"- {col} null values: {null_count} ({null_percentage:.2f}%)")

        # 8. Save results (commented out by default)

        output_path = os.path.join(os.getcwd(), f"{truck_id}_cycle_transformed.csv")
        df_clean.write_csv(output_path)
        print(f"\n💾 Results saved to: {output_path}")

        print(f"\n✅ Cycle ETL Pipeline completed successfully!")
        print(
            f"Final dataset: {df_clean.height} records with {len(df_clean.columns)} columns"
        )

    except Exception as e:
        print(f"\n❌ Critical error in pipeline: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_cycle_etl_pipeline()
