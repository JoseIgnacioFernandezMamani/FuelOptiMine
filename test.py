#!/usr/bin/env python3
"""
Diagnóstico MLflow - Verificar modelos registrados y sus artefactos
"""

import requests
import json
import os
from pathlib import Path

def check_mlflow_models_via_api():
    """Verificar modelos registrados via MLflow API"""
    print("🔍 VERIFICANDO MODELOS REGISTRADOS EN MLFLOW")
    print("=" * 60)
    
    mlflow_url = "http://localhost:5000"
    
    try:
        # Listar todos los modelos registrados
        response = requests.get(f"{mlflow_url}/api/2.0/mlflow/registered-models/list")
        
        if response.status_code == 200:
            models = response.json().get('registered_models', [])
            print(f"Modelos registrados encontrados: {len(models)}")
            
            for model in models:
                name = model['name']
                print(f"\n📦 Modelo: {name}")
                
                # Obtener información detallada del modelo
                versions_response = requests.post(
                    f"{mlflow_url}/api/2.0/mlflow/model-versions/search",
                    json={"name": name}
                )
                
                if versions_response.status_code == 200:
                    versions = versions_response.json().get('model_versions', [])
                    for version in versions:
                        version_num = version['version']
                        source = version.get('source', 'N/A')
                        status = version['status']
                        
                        print(f"   Versión {version_num}:")
                        print(f"     Status: {status}")
                        print(f"     Source: {source}")
                        
                        if '/mnt/d/' in source:
                            print("     ❌ PROBLEMA: Source contiene ruta /mnt/d/")
                        
                        # Verificar el run asociado
                        run_id = version.get('run_id')
                        if run_id:
                            check_run_artifacts(mlflow_url, run_id)
                else:
                    print(f"   Error obteniendo versiones: {versions_response.status_code}")
        else:
            print(f"Error API: {response.status_code}")
            
    except Exception as e:
        print(f"Error consultando MLflow API: {e}")

def check_run_artifacts(mlflow_url, run_id):
    """Verificar artefactos de un run específico"""
    try:
        response = requests.get(f"{mlflow_url}/api/2.0/mlflow/runs/get", 
                              json={"run_id": run_id})
        if response.status_code == 200:
            run_data = response.json().get('run', {})
            artifacts_uri = run_data.get('info', {}).get('artifact_uri', '')
            print(f"     Artifact URI: {artifacts_uri}")
            
            if '/mnt/d/' in artifacts_uri:
                print("     ❌ PROBLEMA: Artifact URI contiene /mnt/d/")
                
    except Exception as e:
        print(f"     Error verificando run: {e}")

def check_model_files_directly():
    """Verificar archivos de modelo directamente en el filesystem"""
    print("\n🔍 VERIFICANDO ARCHIVOS DE MODELO EN FILESYSTEM")
    print("=" * 60)
    
    artifact_path = Path("/home/mina/FuelOptiMine/mlflow_server/artifacts")
    
    if not artifact_path.exists():
        print("❌ No existe el directorio de artefactos")
        return
    
    print(f"Buscando en: {artifact_path}")
    
    # Buscar archivos MLmodel que puedan contener rutas incorrectas
    mlmodel_files = list(artifact_path.glob("**/MLmodel"))
    
    for mlmodel_file in mlmodel_files:
        print(f"\n📄 Revisando: {mlmodel_file.relative_to(artifact_path)}")
        
        try:
            with open(mlmodel_file, 'r') as f:
                content = f.read()
                
            if '/mnt/d/' in content:
                print("❌ PROBLEMA: Archivo MLmodel contiene ruta /mnt/d/")
                # Mostrar líneas problemáticas
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if '/mnt/d/' in line:
                        print(f"   Línea {i}: {line.strip()}")
        except Exception as e:
            print(f"Error leyendo archivo: {e}")

def test_model_loading():
    """Probar cargar un modelo específico"""
    print("\n🔍 PROBANDO CARGA DE MODELOS")
    print("=" * 60)
    
    try:
        import mlflow
        import mlflow.xgboost
        
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Probar con un truck_id específico
        truck_ids = ["T001", "T002", "T003"]  # Ajusta según tus trucks
        
        for truck_id in truck_ids:
            print(f"\nProbando truck: {truck_id}")
            
            try:
                # Stage 4
                model_uri_4 = f"models:/{truck_id}_stage4_fuel/Production"
                print(f"  Stage 4: {model_uri_4}")
                model_4 = mlflow.xgboost.load_model(model_uri_4)
                print("  ✅ Stage 4 cargado correctamente")
            except Exception as e:
                print(f"  ❌ Error Stage 4: {e}")
            
            try:
                # Stage 8
                model_uri_8 = f"models:/{truck_id}_stage8_fuel/Production"
                print(f"  Stage 8: {model_uri_8}")
                model_8 = mlflow.xgboost.load_model(model_uri_8)
                print("  ✅ Stage 8 cargado correctamente")
            except Exception as e:
                print(f"  ❌ Error Stage 8: {e}")
                
    except ImportError:
        print("MLflow no está instalado")
    except Exception as e:
        print(f"Error general: {e}")

def check_database_models():
    """Verificar modelos en la base de datos de MLflow"""
    print("\n🔍 VERIFICANDO BASE DE DATOS MLFLOW")
    print("=" * 60)
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="fuel_optimine",
            user="msc_user2_admin",
            password="msc_user2_password"
        )
        
        cursor = conn.cursor()
        
        # Verificar registered_models
        cursor.execute("SELECT name, source FROM registered_models")
        models = cursor.fetchall()
        
        print("Modelos en registered_models:")
        for name, source in models:
            print(f"  {name}: {source}")
            if source and '/mnt/d/' in source:
                print("    ❌ PROBLEMA: Source contiene /mnt/d/")
        
        # Verificar model_versions
        cursor.execute("SELECT name, version, source FROM model_versions")
        versions = cursor.fetchall()
        
        print("\nVersiones en model_versions:")
        for name, version, source in versions:
            print(f"  {name} v{version}: {source}")
            if source and '/mnt/d/' in source:
                print("    ❌ PROBLEMA: Source contiene /mnt/d/")
        
        conn.close()
        
    except ImportError:
        print("psycopg2 no instalado")
    except Exception as e:
        print(f"Error base de datos: {e}")

def main():
    print("DIAGNÓSTICO MLflow - Modelos y Artefactos")
    print("=" * 60)
    
    check_mlflow_models_via_api()
    check_model_files_directly()
    test_model_loading()
    check_database_models()
    
    print("\n" + "=" * 60)
    print("SOLUCIONES POSIBLES")
    print("=" * 60)
    print("1. LOS MODELOS FUERON REGISTRADOS DESDE UNA MÁQUINA CON RUTA /mnt/d/")
    print("2. Soluciones:")
    print("   a) Re-registrar los modelos desde la máquina actual")
    print("   b) Editar manualmente las rutas en la base de datos MLflow")
    print("   c) Usar enlace simbólico: sudo ln -s /home/mina/FuelOptiMine /mnt/d/Develop/Fueloptimine")
    print("   d) Re-entrenar los modelos en el entorno actual")

if __name__ == "__main__":
    main()