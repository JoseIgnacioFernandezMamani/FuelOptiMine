"""
Promote all truck models to Production stage in MLflow.
Run this after training all models.
"""
import mlflow
from mlflow.tracking import MlflowClient
from mlflow_server.config import MLFLOW_TRACKING_URI, TRUCK_IDS  # ✅ Cambio aquí

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
                    archive_existing_versions=True,
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