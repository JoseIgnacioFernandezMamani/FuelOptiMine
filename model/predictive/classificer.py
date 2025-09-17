import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

# Tus datos de ejemplo
data = {
    "CycleDurationSeconds": [1370, 1007, 663, 650, 600, 1395],
    "FuelConsumed": [1.28, 41.92, 4, 16.32, 10.88, 51.2],
    "Distance": [3659, 3975, 3659, 2414, 1698, 3915],
    "TotalMeasuredTonnage": [0, 168.6, 0, 178.3, 0, 198.1],
}

df = pd.DataFrame(data)


# Rule-based classifier
class RuleBasedClassifier(BaseEstimator):
    def fit(self, X, y=None):
        # No necesita entrenamiento
        return self

    def predict(self, X):
        X = X.reset_index(drop=True)
        prev = None
        results = []
        for _, row in X.iterrows():
            if prev is None:
                results.append(1)  # primer registro bueno
            else:
                fuel_inc = row["FuelConsumed"] > prev["FuelConsumed"]
                dist_inc = row["Distance"] > prev["Distance"]
                tonnage_inc = row["TotalMeasuredTonnage"] > prev["TotalMeasuredTonnage"]
                time_inc = row["CycleDurationSeconds"] > prev["CycleDurationSeconds"]

                # Regla: Fuel increase + al menos una variable relevante aumenta
                results.append(int(fuel_inc and (dist_inc or tonnage_inc or time_inc)))
            prev = row
        return np.array(results)


# Instanciamos y aplicamos
clf = RuleBasedClassifier()
df["is_good"] = clf.predict(df)

print(df)
