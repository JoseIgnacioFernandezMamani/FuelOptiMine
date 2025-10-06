#!/usr/bin/env python3
"""
Promote all truck models to Production stage in MLflow.
Run this after training all models.
"""

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:5000"

TRUCK_IDS = [
    "T-210",
    "T-211",
    "T-212",
    "T-213",
    "T-214",
    "T-215",
    "T-216",
    "T-217",
    "T-218",
    "T-219",
    "T-220",
    "T-221",
    "T-222",
    "T-223",
    "T-224",
    "T-225",
    "T-230",
    "T-231",
    "T-232",
    "T-233",
    "T-236",
    "T-237",
    "T-238",
    "T-240",
    "T-241",
    "T-242",
    "T-243",
]


def promote_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    promoted = 0
    failed = 0

    for truck_id in TRUCK_IDS:
        for stage_name in ["stage4", "stage8"]:
            model_name = f"{truck_id}_{stage_name}_fuel"

            try:
                # Get all versions
                versions = client.search_model_versions(f"name='{model_name}'")

                if not versions:
                    print(f"⚠️  {model_name}: No versions found")
                    failed += 1
                    continue

                # Get latest version number
                latest_version = str(max([int(v.version) for v in versions]))

                # Transition to Production
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True,  # Archive old Production versions
                )

                print(f"✅ {model_name} v{latest_version} → Production")
                promoted += 1

            except Exception as e:
                print(f"❌ {model_name}: {str(e)}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"Promoted: {promoted}")
    print(f"Failed: {failed}")
    print(f"Total: {len(TRUCK_IDS) * 2}")
    print("=" * 60)


if __name__ == "__main__":
    promote_models()
