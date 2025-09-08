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
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8")

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
        .alias("cycle_end"),
        # Medianas móviles
        pl.col("FuelLevelLiters")
        .rolling_median(window_size=10, min_samples=3, center=True)
        .alias("MedianFuelLevelLiters"),
        pl.coalesce("LoadingZone", "Destination").alias("Destination"),
    ]
)

# Crear grupos de ciclos
df = df.with_columns(
    [pl.col("cycle_end").shift(1, fill_value=False).cum_sum().alias("cycle_group")]
)

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
            pl.col("StageSequence").last().alias("StageSequence"),
            pl.col("Destination").last().alias("Destination"),
            pl.len().alias("RecordsInCycle"),
        ]
    )
    .sort("TimeStampIni")
)

result = result.filter((pl.col("Destination").str.strip_chars().str.len_chars() > 2))

result = result.with_columns(
    pl.when((pl.col("StartCycle") - pl.col("EndCycle")).abs() <= 500)
    .then((pl.col("StartCycle") - pl.col("EndCycle")).abs())
    .otherwise(10)
    .alias("FuelConsumed"),
    (pl.col("TimeStampFin") - pl.col("TimeStampIni"))
    .dt.total_seconds()
    .abs()
    .alias("CycleDurationSeconds"),
)

# Limpiar datos
cols_to_clean = [
    "AverageSpeed",
    "AvgSlopePercent",
    "AvgAcceleration",
    "TotalMeasuredTonnage",
    "Distance",
    "FuelConsumed",
]

for col in cols_to_clean:
    result = result.with_columns(
        pl.when(
            pl.col(col).is_infinite() | pl.col(col).is_nan() | pl.col(col).is_null()
        )
        .then(0)
        .otherwise(pl.col(col))
        .alias(col)
    )

print(f"Total de registros procesados: {result.height}")

# SEPARAR DATOS POR STAGESEQUENCE
data_stage_4 = result.filter(pl.col("StageSequence") == 4)
data_stage_8 = result.filter(pl.col("StageSequence") == 8)

# show sumary data
print(f"\nDistribución de datos:")
print(f"Stage 4: {data_stage_4.height} registros")
print(f"Stage 8: {data_stage_8.height} registros")

if data_stage_4.height < 20 or data_stage_8.height < 20:
    print("ERROR: Muy pocos datos para crear modelos robustos")
    exit()

# Variables para el modelo
numeric_features = [
    "Distance",
    "AverageSpeed",
    "AvgSlopePercent",
    "AvgAcceleration",
    "TotalMeasuredTonnage",
]


def prepare_and_train_model(data, stage_name):
    """Preparar datos y entrenar modelo para una etapa específica"""
    print(f"\n=== PROCESANDO STAGE {stage_name} ===")

    # Preparar datos
    X_numeric = data.select(numeric_features).to_pandas()
    X_destination = data.select("Destination").to_pandas()
    y = data.select("FuelConsumed").to_numpy().flatten()

    # Split train/test
    X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_dest_train, X_dest_test = train_test_split(
        X_destination, test_size=0.2, random_state=42, shuffle=True
    )

    # Encoding de destinos
    binary_encoder = ce.BinaryEncoder(cols=["Destination"])
    X_dest_train_encoded = binary_encoder.fit_transform(X_dest_train)
    X_dest_test_encoded = binary_encoder.transform(X_dest_test)

    # Combinar features
    X_train = np.hstack([X_numeric_train.values, X_dest_train_encoded.values])
    X_test = np.hstack([X_numeric_test.values, X_dest_test_encoded.values])

    # Escalado
    scaler = StandardScaler()
    X_numeric_train_scaled = scaler.fit_transform(X_numeric_train)
    X_numeric_test_scaled = scaler.transform(X_numeric_test)

    X_train_scaled = np.hstack([X_numeric_train_scaled, X_dest_train_encoded.values])
    X_test_scaled = np.hstack([X_numeric_test_scaled, X_dest_test_encoded.values])

    # Entrenar modelo
    model = linear_model.LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predicciones
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Métricas
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"R² Train: {train_r2:.4f}")
    print(f"R² Test:  {test_r2:.4f}")
    print(f"MAE Test: {test_mae:.4f}")
    print(f"RMSE Test: {test_rmse:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "encoder": binary_encoder,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mae": test_mae,
        "rmse": test_rmse,
        "feature_names": numeric_features,
    }


# Entrenar modelos
results_4 = prepare_and_train_model(data_stage_4, "4")
results_8 = prepare_and_train_model(data_stage_8, "8")

# CREAR GRÁFICOS COMPLETOS
fig = plt.figure(figsize=(20, 15))

# 1. Métricas Comparativas
ax1 = plt.subplot(3, 4, 1)
metrics = ["R² Train", "R² Test", "MAE", "RMSE"]
stage_4_values = [
    results_4["train_r2"],
    results_4["test_r2"],
    results_4["mae"],
    results_4["rmse"],
]
stage_8_values = [
    results_8["train_r2"],
    results_8["test_r2"],
    results_8["mae"],
    results_8["rmse"],
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(
    x - width / 2, stage_4_values, width, label="Stage 4", alpha=0.8, color="skyblue"
)
bars2 = ax1.bar(
    x + width / 2, stage_8_values, width, label="Stage 8", alpha=0.8, color="lightcoral"
)

ax1.set_xlabel("Métricas")
ax1.set_ylabel("Valores")
ax1.set_title("Comparación de Métricas por Stage")
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# Agregar valores en las barras
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )
for bar in bars2:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.01,
        f"{height:.3f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# 2. Distribución de Residuos - Stage 4
ax2 = plt.subplot(3, 4, 2)
residuals_4 = results_4["y_test"] - results_4["y_test_pred"]
ax2.hist(residuals_4, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
ax2.axvline(
    np.mean(residuals_4),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(residuals_4):.2f}",
)
ax2.set_xlabel("Residuos")
ax2.set_ylabel("Frecuencia")
ax2.set_title("Distribución de Residuos - Stage 4")
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Distribución de Residuos - Stage 8
ax3 = plt.subplot(3, 4, 3)
residuals_8 = results_8["y_test"] - results_8["y_test_pred"]
ax3.hist(residuals_8, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
ax3.axvline(
    np.mean(residuals_8),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(residuals_8):.2f}",
)
ax3.set_xlabel("Residuos")
ax3.set_ylabel("Frecuencia")
ax3.set_title("Distribución de Residuos - Stage 8")
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Q-Q Plot para normalidad de residuos
ax4 = plt.subplot(3, 4, 4)
stats.probplot(residuals_4, dist="norm", plot=ax4)
ax4.set_title("Q-Q Plot Residuos - Stage 4")
ax4.grid(alpha=0.3)

# 5. Predicciones vs Reales - Stage 4
ax5 = plt.subplot(3, 4, 5)
ax5.scatter(results_4["y_test"], results_4["y_test_pred"], alpha=0.6, color="skyblue")
ax5.plot(
    [results_4["y_test"].min(), results_4["y_test"].max()],
    [results_4["y_test"].min(), results_4["y_test"].max()],
    "r--",
    lw=2,
)
ax5.set_xlabel("Valores Reales")
ax5.set_ylabel("Predicciones")
ax5.set_title(f'Predicciones vs Reales - Stage 4\nR² = {results_4["test_r2"]:.3f}')
ax5.grid(alpha=0.3)

# 6. Predicciones vs Reales - Stage 8
ax6 = plt.subplot(3, 4, 6)
ax6.scatter(
    results_8["y_test"], results_8["y_test_pred"], alpha=0.6, color="lightcoral"
)
ax6.plot(
    [results_8["y_test"].min(), results_8["y_test"].max()],
    [results_8["y_test"].min(), results_8["y_test"].max()],
    "r--",
    lw=2,
)
ax6.set_xlabel("Valores Reales")
ax6.set_ylabel("Predicciones")
ax6.set_title(f'Predicciones vs Reales - Stage 8\nR² = {results_8["test_r2"]:.3f}')
ax6.grid(alpha=0.3)

# 7. Detección de Outliers - Stage 4 (Train)


ax7 = plt.subplot(3, 4, 7)
residuals_train_4 = results_4["y_train"] - results_4["y_train_pred"]
z_scores_4 = np.abs(stats.zscore(residuals_train_4))
outliers_4 = z_scores_4 > 2
inliers_4 = z_scores_4 <= 2

indices_4 = np.arange(len(residuals_train_4))
ax7.scatter(
    indices_4[inliers_4],
    residuals_train_4[inliers_4],
    alpha=0.6,
    color="skyblue",
    label=f"Inliers ({np.sum(inliers_4)})",
    s=30,
)
ax7.scatter(
    indices_4[outliers_4],
    residuals_train_4[outliers_4],
    alpha=0.8,
    color="red",
    label=f"Outliers ({np.sum(outliers_4)})",
    s=50,
)
ax7.axhline(0, color="black", linestyle="-", alpha=0.3)
ax7.axhline(2 * np.std(residuals_train_4), color="red", linestyle="--", alpha=0.5)
ax7.axhline(-2 * np.std(residuals_train_4), color="red", linestyle="--", alpha=0.5)
ax7.set_xlabel("Índice de Muestra")
ax7.set_ylabel("Residuos")
ax7.set_title("Inliers vs Outliers - Stage 4 (Train)")
ax7.legend()
ax7.grid(alpha=0.3)

# 8. Detección de Outliers - Stage 8 (Train)
ax8 = plt.subplot(3, 4, 8)
residuals_train_8 = results_8["y_train"] - results_8["y_train_pred"]
z_scores_8 = np.abs(stats.zscore(residuals_train_8))
outliers_8 = z_scores_8 > 2
inliers_8 = z_scores_8 <= 2

# Crear índices correctamente
indices_8 = np.arange(len(residuals_train_8))
ax8.scatter(
    indices_8[inliers_8],
    residuals_train_8[inliers_8],
    alpha=0.6,
    color="lightcoral",
    label=f"Inliers ({np.sum(inliers_8)})",
    s=30,
)
ax8.scatter(
    indices_8[outliers_8],
    residuals_train_8[outliers_8],
    alpha=0.8,
    color="red",
    label=f"Outliers ({np.sum(outliers_8)})",
    s=50,
)
ax8.axhline(0, color="black", linestyle="-", alpha=0.3)
ax8.axhline(2 * np.std(residuals_train_8), color="red", linestyle="--", alpha=0.5)
ax8.axhline(-2 * np.std(residuals_train_8), color="red", linestyle="--", alpha=0.5)
ax8.set_xlabel("Índice de Muestra")
ax8.set_ylabel("Residuos")
ax8.set_title("Inliers vs Outliers - Stage 8 (Train)")
ax8.legend()
ax8.grid(alpha=0.3)

# 9. Distribución del Target por Stage
ax9 = plt.subplot(3, 4, 9)
fuel_4 = data_stage_4.select("FuelConsumed").to_numpy().flatten()
fuel_8 = data_stage_8.select("FuelConsumed").to_numpy().flatten()

ax9.hist(
    fuel_4,
    bins=20,
    alpha=0.6,
    label=f"Stage 4 (μ={np.mean(fuel_4):.1f})",
    color="skyblue",
)
ax9.hist(
    fuel_8,
    bins=20,
    alpha=0.6,
    label=f"Stage 8 (μ={np.mean(fuel_8):.1f})",
    color="lightcoral",
)
ax9.set_xlabel("Consumo de Combustible")
ax9.set_ylabel("Frecuencia")
ax9.set_title("Distribución del Target por Stage")
ax9.legend()
ax9.grid(alpha=0.3)

# 10. Residuos vs Fitted Values - Stage 4
ax10 = plt.subplot(3, 4, 10)
ax10.scatter(results_4["y_test_pred"], residuals_4, alpha=0.6, color="skyblue")
ax10.axhline(0, color="red", linestyle="--")
ax10.set_xlabel("Valores Predichos")
ax10.set_ylabel("Residuos")
ax10.set_title("Residuos vs Fitted - Stage 4")
ax10.grid(alpha=0.3)

# 11. Residuos vs Fitted Values - Stage 8
ax11 = plt.subplot(3, 4, 11)
ax11.scatter(results_8["y_test_pred"], residuals_8, alpha=0.6, color="lightcoral")
ax11.axhline(0, color="red", linestyle="--")
ax11.set_xlabel("Valores Predichos")
ax11.set_ylabel("Residuos")
ax11.set_title("Residuos vs Fitted - Stage 8")
ax11.grid(alpha=0.3)

# 12. Comparación de Performance
ax12 = plt.subplot(3, 4, 12)
performance_data = {
    "Stage 4": [results_4["test_r2"], results_4["mae"], results_4["rmse"]],
    "Stage 8": [results_8["test_r2"], results_8["mae"], results_8["rmse"]],
}
performance_metrics = ["R²", "MAE", "RMSE"]

# Normalizar métricas para comparación visual
norm_data = {}
for stage, values in performance_data.items():
    norm_values = []
    for i, val in enumerate(values):
        if i == 0:  # R² (mayor es mejor)
            norm_values.append(val)
        else:  # MAE, RMSE (menor es mejor, invertir)
            norm_values.append(1 / (1 + val))
    norm_data[stage] = norm_values

x_pos = np.arange(len(performance_metrics))
for i, (stage, values) in enumerate(norm_data.items()):
    ax12.plot(
        x_pos,
        values,
        "o-",
        linewidth=2,
        markersize=8,
        label=stage,
        color=["skyblue", "lightcoral"][i],
    )

ax12.set_xlabel("Métricas")
ax12.set_ylabel("Performance Normalizada")
ax12.set_title("Comparación de Performance\n(Valores normalizados)")
ax12.set_xticks(x_pos)
ax12.set_xticklabels(performance_metrics)
ax12.legend()
ax12.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# RESUMEN FINAL
print("\n" + "=" * 60)
print("RESUMEN DE MODELOS SEPARADOS")
print("=" * 60)
print(f"STAGE 4:")
print(f"  - Registros: {data_stage_4.height}")
print(f"  - R² Test: {results_4['test_r2']:.4f}")
print(f"  - MAE: {results_4['mae']:.4f}")
print(f"  - RMSE: {results_4['rmse']:.4f}")
print(
    f"  - Outliers: {np.sum(outliers_4)} ({100*np.sum(outliers_4)/len(outliers_4):.1f}%)"
)

print(f"\nSTAGE 8:")
print(f"  - Registros: {data_stage_8.height}")
print(f"  - R² Test: {results_8['test_r2']:.4f}")
print(f"  - MAE: {results_8['mae']:.4f}")
print(f"  - RMSE: {results_8['rmse']:.4f}")
print(
    f"  - Outliers: {np.sum(outliers_8)} ({100*np.sum(outliers_8)/len(outliers_8):.1f}%)"
)

avg_r2 = (results_4["test_r2"] + results_8["test_r2"]) / 2
print(f"\nPERFORMANCE PROMEDIO:")
print(f"  - R² Promedio: {avg_r2:.4f}")
print(f"  - Diferencia Consumo Promedio: {np.mean(fuel_8) - np.mean(fuel_4):.2f} L")

print("\n🎯 CONCLUSIÓN:")
if avg_r2 > 0.7:
    print("✅ Excelente performance - Los modelos separados son viables")
elif avg_r2 > 0.5:
    print("⚠️  Performance aceptable - Considerar mejoras en features")
else:
    print("❌ Performance baja - Revisar datos y estrategia de modelado")

# Guardar modelos para uso futuro (opcional)
results_summary = {
    "stage_4": {
        "r2_test": results_4["test_r2"],
        "mae": results_4["mae"],
        "rmse": results_4["rmse"],
        "n_samples": data_stage_4.height,
    },
    "stage_8": {
        "r2_test": results_8["test_r2"],
        "mae": results_8["mae"],
        "rmse": results_8["rmse"],
        "n_samples": data_stage_8.height,
    },
}

with open("model_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\n📊 Resultados guardados en 'model_results.json'")
