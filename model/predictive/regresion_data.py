from polars.selectors import binary
from sklearn.preprocessing._encoders import OneHotEncoder
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
from sklearn.model_selection import train_test_split
import category_encoders as ce
import json

warnings.filterwarnings("ignore")

# Cargar el dataset
df = pl.read_csv("unified_data_T-210.csv", try_parse_dates=True)

df = df.sort("SortTimestamp")

# Identificar registros con ciclo y calcular medianas móviles
df = df.with_columns(
    [
        # Identificar ciclos
        pl.when((pl.col("StageSequence") == 4) | (pl.col("StageSequence") == 8))
        .then(True)
        .otherwise(False)
        .alias("has_cycle"),
        # Medianas móviles
        pl.col("FuelLevelLiters")
        .rolling_median(window_size=10, min_samples=3, center=True)
        .alias("MedianFuelLevelLiters"),
        pl.coalesce("LoadingZone", "Destination").alias("Destination"),
    ]
)

# Crear grupos de ciclos - cada grupo representa un ciclo completo
df = df.with_columns([pl.col("has_cycle").cum_sum().alias("cycle_group")])

result = (
    df.group_by("cycle_group")
    .agg(
        [
            pl.col("TimeStamp").first().alias("TimeStampIni"),
            pl.col("TimeStamp").last().alias("TimeStampFin"),
            pl.col("ShiftDate").last().alias("ShiftDate"),
            pl.col("Equipment").last().alias("Equipment"),
            pl.col("TruckFleet").last().alias("TruckFleet"),
            pl.col("MedianFuelLevelLiters").first().alias("StartCycle"),
            pl.col("MedianFuelLevelLiters").last().alias("EndCycle"),
            pl.col("SpeedAvg").mean().alias("AverageSpeed"),
            pl.col("RPM").mean().alias("AvgRPM"),
            pl.col("SlopePercent").mean().alias("AvgSlopePercent"),
            pl.col("Acceleration").mean().alias("AvgAcceleration"),
            pl.col("MeasuredTonnage").sum().alias("TotalMeasuredTonnage"),
            pl.col("Distance").sum().alias("Distance"),
            pl.col("StageSequence").first().alias("StageSequence"),
            pl.col("Destination").first().alias("Destination"),
            pl.len().alias("RecordsInCycle"),
        ]
    )
    .sort("TimeStampIni")
)

result = result.with_columns(
    pl.when((pl.col("StartCycle") - pl.col("EndCycle")).abs() <= 500)
    .then((pl.col("StartCycle") - pl.col("EndCycle")).abs())
    .otherwise(10)
    .alias("FuelConsumed"),
    (pl.col("TimeStampIni").diff().dt.total_seconds().abs()).alias(
        "CycleDurationSeconds"
    ),
)

result = result.slice(1, result.height - 2)

# columnas que necesitan ser limpiados
cols_to_clean = [
    "AverageSpeed",
    "AvgSlopePercent",
    "AvgAcceleration",
    "TotalMeasuredTonnage",
    "Distance",
    "FuelConsumed",
]

# Limpiar cada columna de valores nulos, NaN o infinitos
for col in cols_to_clean:
    result = result.with_columns(
        pl.when(
            pl.col(col).is_infinite() | pl.col(col).is_nan() | pl.col(col).is_null()
        )
        .then(0)
        .otherwise(pl.col(col))
        .alias(col)
    )

# Guardar datos en el csv de ciclos
result.write_csv("cycle_data.csv")
print("Archivo 'cycle_data.csv' guardado exitosamente")

# Mostrar resumen de los datos procesados
print(f"\nResumen del procesamiento:")
print(f"Total de registros originales: {df.height}")
print(f"Total de ciclos identificados: {result.height}")

if result.height < 10:
    print("ERROR: Muy pocos datos válidos para crear un modelo robusto")
    exit()

# Variables clave para el modelo
numeric_features = [
    "Distance",
    "AverageSpeed",
    "AvgSlopePercent",
    "AvgAcceleration",
    "TotalMeasuredTonnage",
]

# preparar datos numericos
X_numeric = result.select(numeric_features).to_pandas()
X_stage = result.select("StageSequence").to_pandas()
X_destination = result.select("Destination").to_pandas()
y = result.select("FuelConsumed").to_numpy().flatten()

# Split the dataset into training and testing sets
X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X_numeric, y, test_size=0.2, random_state=42
)
X_stage_train, X_stage_test = train_test_split(X_stage, test_size=0.2, random_state=42)
X_destination_train, X_destination_test = train_test_split(
    X_destination, test_size=0.2, random_state=42
)

# Aplicar one-hot encoding a StageSequence
encoder_one_hot = OneHotEncoder(sparse_output=False, drop="first")
X_stage_train_encoded = encoder_one_hot.fit_transform(X_stage_train)
X_stage_test_encoded = encoder_one_hot.transform(X_stage_test)

# Aplicar binary encoding a Destination
binary_encoder = ce.BinaryEncoder(cols=["Destination"])
X_destination_train_encoded = binary_encoder.fit_transform(X_destination_train)
X_destination_test_encoded = binary_encoder.transform(X_destination_test)

# Combinar todas las features
X_train = np.hstack(
    [X_numeric_train.values, X_stage_train_encoded, X_destination_train_encoded.values]
)
X_test = np.hstack(
    [X_numeric_test.values, X_stage_test_encoded, X_destination_test_encoded.values]
)

# Verificar que no hay infinitos ni NaN
x_has_inf = np.isinf(X_train).any()
x_has_nan = np.isnan(X_train).any()
y_has_inf = np.isinf(y_train).any()
y_has_nan = np.isnan(y_train).any()

print(f"X contiene infinitos: {x_has_inf}")
print(f"X contiene NaN: {x_has_nan}")
print(f"Y contiene infinitos: {y_has_inf}")
print(f"Y contiene NaN: {y_has_nan}")

if x_has_inf or x_has_nan or y_has_inf or y_has_nan:
    print("ERROR: Los datos aún contienen valores inválidos después de la limpieza")
    exit()

# Escalar solo las variables numéricas
scaler = StandardScaler()
X_numeric_train_scaled = scaler.fit_transform(X_numeric_train)
X_numeric_test_scaled = scaler.transform(X_numeric_test)

# Combinar features escaladas
X_train_scaled = np.hstack(
    [X_numeric_train_scaled, X_stage_train_encoded, X_destination_train_encoded.values]
)
X_test_scaled = np.hstack(
    [X_numeric_test_scaled, X_stage_test_encoded, X_destination_test_encoded.values]
)

# Verificar que el escalado no introdujo problemas
x_scaled_has_inf = np.isinf(X_train_scaled).any()
x_scaled_has_nan = np.isnan(X_train_scaled).any()

print(f"X escalado contiene infinitos: {x_scaled_has_inf}")
print(f"X escalado contiene NaN: {x_scaled_has_nan}")
print(f"Rango de X escalado: {X_train_scaled.min():.2f} - {X_train_scaled.max():.2f}")

if x_scaled_has_inf or x_scaled_has_nan:
    print("ERROR: El proceso de escalado introdujo valores inválidos")
    exit()

# Modelo de regresión lineal
lr = linear_model.LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predecir y evaluar el modelo lineal
y_test_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_test_pred_lr)
r2_lr = r2_score(y_test, y_test_pred_lr)
print(f"Linear Regression - Mean Squared Error: {mse_lr:.4f}")
print(f"Linear Regression - R² Score: {r2_lr:.4f}")

# ============ OPTIMIZACIÓN RANSAC ============


def optimize_ransac_multivariable(x, y, n_seeds=25):
    """
    Optimiza RANSAC para modelos multivariables
    """
    n_samples, n_features = x.shape

    # Configuraciones adaptadas al tamaño de los datos
    configurations = [
        # Configuración conservadora
        {
            "min_samples": max(n_features + 1, n_samples // 10),
            "residual_threshold": None,
            "max_trials": 300,
            "stop_score": 0.85,
            "stop_probability": 0.90,
            "loss": "absolute_error",
        },
        # Configuración permisiva
        {
            "min_samples": max(n_features + 1, n_samples // 15),
            "residual_threshold": None,
            "max_trials": 500,
            "stop_score": 0.80,
            "stop_probability": 0.95,
            "loss": "absolute_error",
        },
        # Configuración muy permisiva
        {
            "min_samples": max(n_features + 1, n_samples // 20),
            "residual_threshold": None,
            "max_trials": 800,
            "stop_score": 0.75,
            "stop_probability": 0.95,
            "loss": "absolute_error",
        },
    ]

    best_score = -np.inf
    best_model = None
    best_params = None
    best_metrics = {}

    print("Optimizando RANSAC para modelo multivariable...")
    total_tests = len(configurations) * n_seeds
    current_test = 0

    for config_idx, config in enumerate(configurations):
        print(f"Probando configuración {config_idx + 1}/{len(configurations)}")

        for seed in range(n_seeds):
            current_test += 1

            try:
                params = config.copy()
                params["random_state"] = seed

                ransac = linear_model.RANSACRegressor(**params)
                ransac.fit(x, y)

                # Calcular métricas
                y_pred = ransac.predict(x)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)

                # Métricas específicas de RANSAC
                inlier_mask = ransac.inlier_mask_
                n_inliers = np.sum(inlier_mask)
                inlier_ratio = n_inliers / len(y)

                # R² en inliers
                if n_inliers > n_features + 1:
                    r2_inliers = r2_score(y[inlier_mask], y_pred[inlier_mask])
                else:
                    r2_inliers = -np.inf

                # Score compuesto
                composite_score = (
                    0.6 * max(0, r2)
                    + 0.2 * max(0, r2_inliers)
                    + 0.15 * inlier_ratio
                    + 0.05 * max(0, 1 - mae / (np.max(y) - np.min(y)))
                )

                if composite_score > best_score and r2 > 0:
                    best_score = composite_score
                    best_model = ransac
                    best_params = params.copy()
                    best_metrics = {
                        "r2": r2,
                        "r2_inliers": r2_inliers,
                        "mae": mae,
                        "n_inliers": n_inliers,
                        "inlier_ratio": inlier_ratio,
                        "composite_score": composite_score,
                        "n_trials": ransac.n_trials_,
                    }

                if current_test % 10 == 0:
                    progress = (current_test / total_tests) * 100
                    best_r2 = best_metrics.get("r2", 0)
                    print(f"Progreso: {progress:.1f}% - Mejor R²: {best_r2:.4f}")

            except Exception as e:
                print(f"Error en iteración {current_test}: {str(e)}")
                continue

    return best_model, best_params, best_metrics


# Ejecutar optimización RANSAC
print("\n=== INICIANDO OPTIMIZACIÓN DE RANSAC ===")
try:
    ransac_optimized, best_params, best_metrics = optimize_ransac_multivariable(
        X_train_scaled, y_train, n_seeds=15
    )

    if ransac_optimized is not None:
        ransac = ransac_optimized
        print(f"\n=== MEJORES PARÁMETROS ENCONTRADOS ===")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        print(f"\n=== MEJORES MÉTRICAS ===")
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    else:
        raise ValueError("No se pudo encontrar una configuración RANSAC válida")

except Exception as e:
    print(f"Error en optimización RANSAC: {e}")
    print("Usando configuración RANSAC por defecto...")

    # Configuración fallback más simple
    min_samples = max(X_train_scaled.shape[1] + 1, len(X_train_scaled) // 20)

    ransac = linear_model.RANSACRegressor(
        random_state=42,
        min_samples=min_samples,
        residual_threshold=np.std(y_train),
        max_trials=500,
        stop_probability=0.95,
        loss="absolute_error",
    )
    ransac.fit(X_train_scaled, y_train)

# Predicciones RANSAC en test
y_test_pred_ransac = ransac.predict(X_test_scaled)

# Predicciones de regresión lineal en test
y_test_pred_lr = lr.predict(X_test_scaled)

# Métricas en test
mse_lr = mean_squared_error(y_test, y_test_pred_lr)
r2_lr = r2_score(y_test, y_test_pred_lr)
mse_ransac = mean_squared_error(y_test, y_test_pred_ransac)
r2_ransac = r2_score(y_test, y_test_pred_ransac)

print(f"Linear Regression (Test) - MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}")
print(f"RANSAC (Test) - MSE: {mse_ransac:.4f}, R²: {r2_ransac:.4f}")

# Predicciones en train para visualización
y_train_pred_lr = lr.predict(X_train_scaled)
y_train_pred_ransac = ransac.predict(X_train_scaled)

# Obtener máscaras de inliers/outliers en train
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# GRÁFICAS MEJORADAS
plt.figure(figsize=(18, 12))

# 1. Gráfica de Valores Reales vs Predicciones (Regresión Lineal) - TEST
plt.subplot(2, 3, 1)
plt.scatter(y_test, y_test_pred_lr, alpha=0.6, color="navy", s=40)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Línea perfecta",
)
plt.xlabel("Valores Reales (Litros)")
plt.ylabel("Predicciones LR (Litros)")
plt.title(f"Regresión Lineal (Test)\nR² = {r2_lr:.3f}")
plt.grid(True, alpha=0.3)
plt.legend()

# 2. Gráfica de Valores Reales vs Predicciones (RANSAC) - TEST
plt.subplot(2, 3, 2)
plt.scatter(y_test, y_test_pred_ransac, alpha=0.7, color="cornflowerblue", s=40)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Línea perfecta",
)
plt.xlabel("Valores Reales (Litros)")
plt.ylabel("Predicciones RANSAC (Litros)")
plt.title(f"RANSAC Optimizado (Test)\nR² = {r2_ransac:.3f}")
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Comparación de Residuos en TEST
plt.subplot(2, 3, 3)
residuals_lr_test = y_test - y_test_pred_lr
residuals_ransac_test = y_test - y_test_pred_ransac

plt.scatter(
    y_test_pred_lr, residuals_lr_test, alpha=0.6, color="navy", s=30, label="LR"
)
plt.scatter(
    y_test_pred_ransac,
    residuals_ransac_test,
    alpha=0.6,
    color="cornflowerblue",
    s=30,
    label="RANSAC",
)
plt.axhline(y=0, color="r", linestyle="--", alpha=0.8)
plt.xlabel("Predicciones (Litros)")
plt.ylabel("Residuos (Litros)")
plt.title("Comparación de Residuos (Test)")
plt.grid(True, alpha=0.3)
plt.legend()

# 4. Distribución de Residuos en TEST
plt.subplot(2, 3, 4)
plt.hist(residuals_lr_test, bins=20, alpha=0.6, color="navy", label="LR", density=True)
plt.hist(
    residuals_ransac_test,
    bins=20,
    alpha=0.6,
    color="cornflowerblue",
    label="RANSAC",
    density=True,
)
plt.xlabel("Residuos (Litros)")
plt.ylabel("Densidad")
plt.title("Distribución de Residuos (Test)")
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Inliers vs Outliers en TRAIN
plt.subplot(2, 3, 5)
plt.scatter(
    y_train[inlier_mask],
    y_train_pred_ransac[inlier_mask],
    alpha=0.7,
    color="yellowgreen",
    s=40,
    label=f"Inliers ({np.sum(inlier_mask)})",
)
plt.scatter(
    y_train[outlier_mask],
    y_train_pred_ransac[outlier_mask],
    alpha=0.7,
    color="gold",
    s=40,
    label=f"Outliers ({np.sum(outlier_mask)})",
)
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "r--",
    lw=2,
    label="Línea perfecta",
)
plt.xlabel("Valores Reales (Litros)")
plt.ylabel("Predicciones RANSAC (Litros)")
plt.title("RANSAC: Inliers vs Outliers (Train)")
plt.grid(True, alpha=0.3)
plt.legend()

# 6. Métricas comparativas
plt.subplot(2, 3, 6)
metrics_names = ["R²", "MAE", "RMSE"]
lr_metrics = [r2_lr, mean_absolute_error(y_test, y_test_pred_lr), np.sqrt(mse_lr)]
ransac_metrics = [
    r2_ransac,
    mean_absolute_error(y_test, y_test_pred_ransac),
    np.sqrt(mse_ransac),
]

x_pos = np.arange(len(metrics_names))
width = 0.35

plt.bar(
    x_pos - width / 2,
    lr_metrics,
    width,
    label="Regresión Lineal",
    color="navy",
    alpha=0.7,
)
plt.bar(
    x_pos + width / 2,
    ransac_metrics,
    width,
    label="RANSAC",
    color="cornflowerblue",
    alpha=0.7,
)

plt.xlabel("Métricas")
plt.ylabel("Valores")
plt.title("Comparación de Métricas (Test)")
plt.xticks(x_pos, metrics_names)
plt.legend()
plt.grid(True, alpha=0.3)

# Añadir valores en las barras
for i, (lr_val, ransac_val) in enumerate(zip(lr_metrics, ransac_metrics)):
    plt.text(
        i - width / 2,
        lr_val + max(lr_metrics) * 0.01,
        f"{lr_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    plt.text(
        i + width / 2,
        ransac_val + max(ransac_metrics) * 0.01,
        f"{ransac_val:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.suptitle(
    "Análisis Completo: Regresión Lineal vs RANSAC con Variables Categóricas",
    y=1.02,
    fontsize=16,
)
plt.show()

# Mostrar métricas finales
print(f"\n=== MÉTRICAS DEL MODELO FINAL (TEST) ===")
print(f"Número de iteraciones realizadas: {ransac.n_trials_}")
print(f"Número de inliers (train): {np.sum(inlier_mask)}")
print(f"Número de outliers (train): {np.sum(outlier_mask)}")
print(f"Porcentaje de inliers: {np.sum(inlier_mask)/len(inlier_mask)*100:.1f}%")

print(f"\n=== COMPARACIÓN DETALLADA DE MODELOS (TEST) ===")
print(f"R² Regresión Lineal: {r2_lr:.4f}")
print(f"R² RANSAC Optimizado: {r2_ransac:.4f}")
print(f"Mejora en R²: {r2_ransac - r2_lr:.4f}")

print(f"\nMAE Regresión Lineal: {mean_absolute_error(y_test, y_test_pred_lr):.4f}")
print(f"MAE RANSAC: {mean_absolute_error(y_test, y_test_pred_ransac):.4f}")

print(f"\nRMSE Regresión Lineal: {np.sqrt(mse_lr):.4f}")
print(f"RMSE RANSAC: {np.sqrt(mse_ransac):.4f}")

# Mostrar importancia de características
n_numeric = len(numeric_features)
n_stage = X_stage_train_encoded.shape[1]
n_destination = X_destination_train_encoded.shape[1]

feature_names = (
    numeric_features
    + [f"StageSequence_{i}" for i in range(n_stage)]
    + [f"Destination_bin_{i}" for i in range(n_destination)]
)

print(f"\n=== IMPORTANCIA DE CARACTERÍSTICAS ===")
lr_coef = np.abs(lr.coef_)
ransac_coef = np.abs(ransac.estimator_.coef_)

print("Top 10 características más importantes:")
lr_importance = list(zip(feature_names, lr_coef))
ransac_importance = list(zip(feature_names, ransac_coef))

lr_importance.sort(key=lambda x: x[1], reverse=True)
ransac_importance.sort(key=lambda x: x[1], reverse=True)

for i in range(min(10, len(feature_names))):
    lr_feat, lr_val = lr_importance[i]
    ransac_feat, ransac_val = ransac_importance[i]
    print(f"LR #{i+1}: {lr_feat:25s} = {lr_val:.4f}")
    print(f"RANSAC #{i+1}: {ransac_feat:25s} = {ransac_val:.4f}")
    print()

# Guardar resultados con predicciones
result_train = result.slice(0, len(y_train))  # Mantener solo datos de entrenamiento
result_with_predictions = result_train.with_columns(
    [
        pl.Series("is_inlier", inlier_mask),
        pl.Series("is_outlier", outlier_mask),
        pl.Series("predicted_fuel_lr", y_train_pred_lr),
        pl.Series("predicted_fuel_ransac", y_train_pred_ransac),
        pl.Series("residual_lr", y_train - y_train_pred_lr),
        pl.Series("residual_ransac", y_train - y_train_pred_ransac),
    ]
)

result_with_predictions.write_csv("predictions_with_categorical.csv")
print("\nArchivo 'predictions_with_categorical.csv' guardado exitosamente")

# Guardar parámetros del modelo si la optimización fue exitosa
if "best_params" in locals() and best_params is not None:
    with open("best_ransac_params_categorical.json", "w") as f:
        json_params = {}
        for k, v in best_params.items():
            if isinstance(v, np.integer):
                json_params[k] = int(v)
            elif isinstance(v, np.floating):
                json_params[k] = float(v)
            else:
                json_params[k] = v
        json.dump(json_params, f, indent=2)
    print("Mejores parámetros guardados en 'best_ransac_params_categorical.json'")

print("\n=== ANÁLISIS COMPLETADO EXITOSAMENTE ===")
print(f"Dimensiones finales: {X_train_scaled.shape[1]} variables independientes")
print(f"  - {len(numeric_features)} variables numéricas")
print(f"  - {n_stage} variables de StageSequence (OneHot)")
print(f"  - {n_destination} variables de Destination (Binary)")
