from etl_core.transform.implementation.time_model.transformer import (
    TimeModelTransformer,
)
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
import sys
import os
import polars as pl


def run_timemodel_etl_pipeline():
    # Configuración de rutas y parámetros
    truck_id = "T-211"
    dataset_name = "train_data"
    data_type = "time_model"  # Cambiado a timemodel

    print(f"🚚 Starting ETL for truck {truck_id} - {dataset_name} (Time Model Data)")

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
        df_raw = raw_data[data_type]
        print(f"✅ Raw data loaded: {df_raw.height} records")
        print("Initial schema:", df_raw.schema)

        # 2. Data transformation
        print("\n🔄 Processing data with TimeModelTransformer...")
        transformer = TimeModelTransformer()
        df_clean = transformer.run_transform(df_raw)

        if df_clean is None or df_clean.is_empty():
            raise RuntimeError("Transformation returned empty data")

        # 3. Verify expected columns
        expected_columns = [
            "ShiftDate",
            "Shift",
            "TimeStamp",
            "RecordDuration",
            "Equipment",
            "TruckFleet",
            "Status",
            "Category",
            "Event",
        ]

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
            ("Categorical empty fixed", "categorical_empty_fixed"),
            ("Negative durations fixed", "negative_durations_fixed"),
            ("Invalid status combinations", "invalid_status_combinations"),
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
            "TimeStamp",
            "Status",
            "Category",
            "Event",
            "RecordDuration",
        ]
        print(df_clean.select(sample_columns).head(5))

        # 6. Save results

        output_path = os.path.join(os.getcwd(), f"{truck_id}_timemodel_transformed.csv")
        df_clean.write_csv(output_path)
        print(f"\n💾 Results saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Critical error in pipeline: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_timemodel_etl_pipeline()
