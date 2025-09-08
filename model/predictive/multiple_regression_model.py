from sklearn.model_selection import train_test_split
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def multiple_regression_model(data, predictor_variables, stage_name):
    """
    Construir y entrenar un modelo de regresión lineal múltiple
    para predecir el consumo de combustible basado en múltiples variables predictoras
    """
    print(f"\n=== MODELO DE REGRESIÓN LINEAL MÚLTIPLE - STAGE {stage_name} ===")
    print(
        f"Variables predictoras: {len(predictor_variables)} numéricas + Destination (categórica)"
    )
    print(f"Variable dependiente: FuelConsumed")

    # Preparar variables predictoras numéricas
    X_numeric = data.select(predictor_variables).to_pandas()
    X_destination = data.select("Destination").to_pandas()
    y = data.select("FuelConsumed").to_numpy().flatten()  # Variable dependiente

    # División entrenamiento/prueba
    X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_dest_train, X_dest_test = train_test_split(
        X_destination, test_size=0.2, random_state=42, shuffle=True
    )

    # Codificación binaria para variable categórica 'Destination'
    binary_encoder = ce.BinaryEncoder(cols=["Destination"])
    X_dest_train_encoded = binary_encoder.fit_transform(X_dest_train)
    X_dest_test_encoded = binary_encoder.transform(X_dest_test)

    # Combinar variables numéricas y categóricas codificadas
    X_train = np.hstack([X_numeric_train.values, X_dest_train_encoded.values])
    X_test = np.hstack([X_numeric_test.values, X_dest_test_encoded.values])

    # Estandarización de variables numéricas (importante para regresión múltiple)
    scaler = StandardScaler()
    X_numeric_train_scaled = scaler.fit_transform(X_numeric_train)
    X_numeric_test_scaled = scaler.transform(X_numeric_test)

    # Matriz final de características escaladas
    X_train_scaled = np.hstack([X_numeric_train_scaled, X_dest_train_encoded.values])
    X_test_scaled = np.hstack([X_numeric_test_scaled, X_dest_test_encoded.values])

    # Entrenar modelo de regresión lineal múltiple
    multiple_regression_model = linear_model.LinearRegression()
    multiple_regression_model.fit(X_train_scaled, y_train)

    # Realizar predicciones
    y_train_pred = multiple_regression_model.predict(X_train_scaled)
    y_test_pred = multiple_regression_model.predict(X_test_scaled)

    # Calcular métricas de evaluación
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"R² Entrenamiento: {train_r2:.4f}")
    print(f"R² Prueba:        {test_r2:.4f}")
    print(f"MAE Prueba:       {test_mae:.4f} litros")
    print(f"RMSE Prueba:      {test_rmse:.4f} litros")

    # Mostrar ecuación del modelo
    print(f"\nEcuación del modelo (variables estandarizadas):")
    print(f"FuelConsumed = {multiple_regression_model.intercept_:.4f}", end="")
    for i, coef in enumerate(
        multiple_regression_model.coef_[: len(predictor_variables)]
    ):
        print(f" + ({coef:.4f} × {predictor_variables[i]})", end="")
    print(f" + términos_destination")

    # Retornar diccionario con todos los componentes del modelo
    return {
        "multiple_regression_model": multiple_regression_model,  # Modelo entrenado
        "feature_scaler": scaler,  # Escalador de características
        "destination_encoder": binary_encoder,  # Codificador de destinos
        "training_targets": y_train,  # Valores reales entrenamiento
        "test_targets": y_test,  # Valores reales prueba
        "training_predictions": y_train_pred,  # Predicciones entrenamiento
        "test_predictions": y_test_pred,  # Predicciones prueba
        "r2_training": train_r2,  # Coeficiente determinación (entrenamiento)
        "r2_test": test_r2,  # Coeficiente determinación (prueba)
        "mean_absolute_error": test_mae,  # Error absoluto medio
        "root_mean_squared_error": test_rmse,  # Raíz error cuadrático medio
        "predictor_variables": predictor_variables,  # Nombres variables predictoras
        "total_features": X_train_scaled.shape[1],  # Número total de características
    }
