"""MLflow configuration for FuelOptiMine models"""

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

# Features configuration
NUMERIC_PREDICTOR_VARS = [
    "SpeedAvg",
    "TotalMeasuredTonnage",
    "Distance",
    "CycleDurationSeconds",
    "StageSequence",
    "TimeEfficiencyPercentage",
]

CATEGORICAL_VARS = ["Destination", "DestinationType", "Material", "Shovel"]
