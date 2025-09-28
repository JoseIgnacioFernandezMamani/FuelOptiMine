import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# ==========================
# 1. Datos
# ==========================
y = np.array(
    [
        1015.67,
        994.23,
        991.67,
        994.23,
        999.67,
        991.67,
        988.79,
        991.67,
        978.23,
        986.23,
        999.67,
        986.23,
        994.23,
        1015.67,
        994.23,
        1015.67,
    ]
)

# Variable independiente ordinal (1, 2, 3, ..., n)
X = np.arange(1, len(y) + 1).reshape(-1, 1)

# ==========================
# 2. Ajuste del modelo RANSAC
# ==========================
ransac = linear_model.TheilSenRegressor(
    fit_intercept=True,
    copy_X=True,
    max_subpopulation=1e4,
    n_subsamples=None,
    max_iter=300,
    tol=1e-3,
    random_state=42,
)
ransac.fit(X, y)

# ==========================
# 3. Predicciones específicas
# ==========================
# Predicción en el primer valor (entrada = 1)
pred_inicio = ransac.predict(np.array([[1]]))[0]

# Predicción en el último valor (entrada = 17)
pred_fin = ransac.predict(np.array([[len(y)]]))[0]

# Diferencia entre ambas predicciones
diferencia = pred_inicio - pred_fin

# ==========================
# 4. Mostrar resultados
# ==========================
print("📈 Predicción RANSAC para entrada 1:     ", round(pred_inicio, 2))
print("📉 Predicción RANSAC para entrada 17:    ", round(pred_fin, 2))
print("🔁 Diferencia (inicio - fin):            ", round(diferencia, 2))

# ==========================
# (Opcional) Visualización
# ==========================
line_X = np.arange(1, len(y) + 1).reshape(-1, 1)
line_y_ransac = ransac.predict(line_X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="gray", label="Datos originales")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=2,
    label="RANSAC Regression",
)
plt.scatter(
    [1, len(y)], [pred_inicio, pred_fin], color="red", zorder=5, label="Predicciones"
)
plt.title("Predicción con RANSAC")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()
