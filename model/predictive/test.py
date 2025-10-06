#!/usr/bin/env python3
import os
import sys
import subprocess
import pandas as pd

# ========== Asegurar que el directorio raíz esté en sys.path ==========
# Esto es CRUCIAL para que los imports funcionen en subprocess
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========== CONFIGURACIÓN GLOBAL ==========
NUMERIC_VARS = [
    "SpeedAvg",
    "TotalMeasuredTonnage",
    "Distance",
    "CycleDurationSeconds",
    "StageSequence",
    "TimeEfficiencyPercentage",
]

CATEGORICAL_VARS = ["Destination", "DestinationType", "Material", "Shovel"]

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

OUTPUT_DIR = "predictions_temp"
FINAL_OUTPUT = "all_predictions.csv"


# ========== FUNCIÓN DE TRABAJO (se ejecuta en subprocess) ==========
def process_truck_in_isolation(truck_id: str, output_dir: str):
    """Esta función se ejecuta en un proceso hijo aislado."""
    from model.predictive.xgboost_model import XGBoostModel

    print(f"🚛 Procesando camión: {truck_id}", flush=True)

    model = XGBoostModel(
        truck_id=truck_id,
        numeric_predictor_vars=NUMERIC_VARS,
        categorical_vars=CATEGORICAL_VARS,
        max_cat_to_onehot=4,
    )

    print("📥 Cargando datos...", flush=True)
    model.load_data()

    print("🔄 Transformando datos...", flush=True)
    model.transform_cycles_data()

    print("🏋️‍♂️ Entrenando modelo...", flush=True)
    model.train()

    print("🔮 Generando predicciones...", flush=True)
    df_pred = model.get_predictions()

    output_path = os.path.join(output_dir, f"{truck_id}_predictions.csv")
    df_pred.write_csv(output_path)
    print(output_path, flush=True)


# ========== FUNCIÓN PARA AGREGAR AL ARCHIVO FINAL ==========
def append_to_final_csv(temp_csv: str, final_csv: str):
    df_temp = pd.read_csv(temp_csv)
    if "TruckID" not in df_temp.columns:
        truck_id = os.path.basename(temp_csv).replace("_predictions.csv", "")
        df_temp["TruckID"] = truck_id

    if not os.path.exists(final_csv):
        df_temp.to_csv(final_csv, index=False)
    else:
        df_temp.to_csv(final_csv, mode="a", header=False, index=False)


# ========== FUNCIÓN PRINCIPAL (orquestador) ==========
def main_orchestrator():
    print("=" * 80)
    print("🚀 Pipeline de Predicción por Camión (procesos aislados - un solo archivo)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(FINAL_OUTPUT):
        os.remove(FINAL_OUTPUT)

    for i, truck_id in enumerate(TRUCK_IDS, 1):
        print(f"\n[{i}/{len(TRUCK_IDS)}] 🔄 Iniciando proceso aislado para: {truck_id}")

        result = subprocess.run(
            [sys.executable, __file__, "--worker", truck_id, OUTPUT_DIR],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"\n❌ Error en {truck_id}:\n{result.stderr}")
            sys.exit(1)

        lines = result.stdout.strip().split("\n")
        temp_csv = lines[-1].strip() if lines else None

        if not temp_csv or not os.path.exists(temp_csv):
            print(f"\n❌ No se generó archivo para {truck_id}")
            sys.exit(1)

        append_to_final_csv(temp_csv, FINAL_OUTPUT)
        os.remove(temp_csv)
        print(f"✅ Completado: {truck_id}")

    print("\n" + "=" * 80)
    print("🎉 ¡Proceso finalizado con éxito!")
    print(f"📁 Archivo final: {os.path.abspath(FINAL_OUTPUT)}")
    print("=" * 80)


# ========== PUNTO DE ENTRADA ==========
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        if len(sys.argv) != 4:
            print("Uso interno: --worker <truck_id> <output_dir>")
            sys.exit(1)
        truck_id = sys.argv[2]
        output_dir = sys.argv[3]
        process_truck_in_isolation(truck_id, output_dir)
    else:
        main_orchestrator()
