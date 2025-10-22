"""
MLflow Training Pipeline for XGBoost Fuel Consumption Models

This module trains specialized XGBoost models for multiple trucks and logs them to MLflow.
Each truck gets two models: one for Stage 4 (empty truck) and one for Stage 8 (loaded truck).

Author: FuelOptiMine Team
Date: 2025-10-04
"""

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from .xgboost_model import XGBoostModel
from mlflow_server.config import (
    MLFLOW_TRACKING_URI,
    TRUCK_IDS,
    NUMERIC_PREDICTOR_VARS,
    CATEGORICAL_VARS
)
import logging
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MLflowTrainer")


def train_and_log_truck_model(truck_id: str) -> bool:
    """
    Train XGBoost models for a single truck and log to MLflow.

    This function trains two specialized models:
    - Stage 4: Empty truck returning to loading zone
    - Stage 8: Loaded truck traveling to dump site

    Args:
        truck_id: Truck identifier (e.g., "T-210")

    Returns:
        bool: True if training succeeded, False otherwise
    """

    # Connect to MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{truck_id}_fuel_prediction")

    try:
        logger.info(f"Starting training for {truck_id}")

        # Define features for each stage model
        features_stage4_numeric = [
            "SpeedAvg",
            "Distance",
            "CycleDurationSeconds",
            "TimeEfficiencyPercentage",
        ]
        features_stage4_categorical = [
            "Shovel",
            "Destination",
            "DestinationType",
        ]

        features_stage8_numeric = [
            "SpeedAvg",
            "TotalMeasuredTonnage",
            "Distance",
            "CycleDurationSeconds",
            "TimeEfficiencyPercentage",
        ]
        features_stage8_categorical = ["Destination", "DestinationType", "Material"]

        # Start MLflow run
        with mlflow.start_run(
            run_name=f"{truck_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Initialize model
            model = XGBoostModel(
                truck_id=truck_id,
                numeric_predictor_vars=NUMERIC_PREDICTOR_VARS,
                categorical_vars=CATEGORICAL_VARS,
                max_cat_to_onehot=4,
            )

            # Load and transform data
            model.load_data()
            model.transform_cycles_data()

            # Train both stage models
            results = model.train()

            # Prepare data samples for signature inference
            df = model.cycles_data.to_pandas()
            df_stage4 = df[df["StageSequence"] == 4]
            df_stage8 = df[df["StageSequence"] == 8]

            # Sample Stage 4 data
            sample_stage4_df = df_stage4.sample(
                n=min(10, len(df_stage4)), random_state=42
            )[features_stage4_numeric + features_stage4_categorical].copy()

            # Sample Stage 8 data
            sample_stage8_df = df_stage8.sample(
                n=min(10, len(df_stage8)), random_state=42
            )[features_stage8_numeric + features_stage8_categorical].copy()

            # Convert numeric columns to float
            for col in features_stage4_numeric:
                if col in sample_stage4_df.columns:
                    sample_stage4_df[col] = sample_stage4_df[col].astype(float)

            for col in features_stage8_numeric:
                if col in sample_stage8_df.columns:
                    sample_stage8_df[col] = sample_stage8_df[col].astype(float)

            # Convert categorical columns to 'category' dtype (required by XGBoost)
            for col in features_stage4_categorical:
                if col in sample_stage4_df.columns:
                    sample_stage4_df[col] = sample_stage4_df[col].astype("category")

            for col in features_stage8_categorical:
                if col in sample_stage8_df.columns:
                    sample_stage8_df[col] = sample_stage8_df[col].astype("category")

            # Generate predictions for signature inference
            predictions_stage4 = model.model_stage4.predict(sample_stage4_df)
            predictions_stage8 = model.model_stage8.predict(sample_stage8_df)

            # Infer model signatures
            signature_stage4 = infer_signature(sample_stage4_df, predictions_stage4)
            signature_stage8 = infer_signature(sample_stage8_df, predictions_stage8)

            # Extract metrics from results
            stage4_metrics = results["stage4"]["metrics"]
            stage8_metrics = results["stage8"]["metrics"]

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "stage4_R2": stage4_metrics["R2"],
                    "stage4_MAE": stage4_metrics["MAE"],
                    "stage4_RMSE": stage4_metrics["RMSE"],
                    "stage4_MAPE": stage4_metrics["MAPE_Safe"],
                    "stage4_RMSLE": stage4_metrics["RMSLE"],
                    "stage8_R2": stage8_metrics["R2"],
                    "stage8_MAE": stage8_metrics["MAE"],
                    "stage8_RMSE": stage8_metrics["RMSE"],
                    "stage8_MAPE": stage8_metrics["MAPE_Safe"],
                    "stage8_RMSLE": stage8_metrics["RMSLE"],
                }
            )

            # Log hyperparameters and training metadata
            mlflow.log_params(
                {
                    "truck_id": truck_id,
                    "numeric_vars_count": len(NUMERIC_PREDICTOR_VARS),
                    "categorical_vars_count": len(CATEGORICAL_VARS),
                    "max_cat_to_onehot": 4,
                    "train_samples_stage4": results["stage4"]["samples"]["train"],
                    "test_samples_stage4": results["stage4"]["samples"]["test"],
                    "train_samples_stage8": results["stage8"]["samples"]["train"],
                    "test_samples_stage8": results["stage8"]["samples"]["test"],
                    "total_consumed_fuel": results["total_consumed_fuel"],
                }
            )

            # Log Stage 4 model (empty truck)
            mlflow.xgboost.log_model(
                model.model_stage4,
                name="model_stage4",
                signature=signature_stage4,
                # input_example=input_example_stage4,
                registered_model_name=f"{truck_id}_stage4_fuel",
            )

            # Log Stage 8 model (loaded truck)
            mlflow.xgboost.log_model(
                model.model_stage8,
                name="model_stage8",
                signature=signature_stage8,
                # input_example=input_example_stage8,
                registered_model_name=f"{truck_id}_stage8_fuel",
            )

            logger.info(
                f"Successfully trained {truck_id} - "
                f"RMSE Stage4: {stage4_metrics['RMSE']:.4f}, "
                f"Stage8: {stage8_metrics['RMSE']:.4f}"
            )
            return True

    except Exception as e:
        logger.error(f"Training failed for {truck_id}: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """
    Main execution function - trains models for all trucks sequentially.
    """
    print("=" * 80)
    print("MLflow Batch Training Pipeline - XGBoost Fuel Models")
    print("=" * 80)

    successful = 0
    failed = 0
    failed_trucks = []

    # Train each truck sequentially
    for i, truck_id in enumerate(TRUCK_IDS, 1):
        print(f"\n[{i}/{len(TRUCK_IDS)}] Processing {truck_id}...")

        success = train_and_log_truck_model(truck_id)

        if success:
            successful += 1
        else:
            failed += 1
            failed_trucks.append(truck_id)

    # Print summary report
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total trucks: {len(TRUCK_IDS)}")
    print(f"Successful: {successful}/{len(TRUCK_IDS)}")
    print(f"Failed: {failed}")

    if failed_trucks:
        print(f"\nFailed trucks: {', '.join(failed_trucks)}")

    print("=" * 80)


if __name__ == "__main__":
    main()
