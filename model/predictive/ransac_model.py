from sklearn.model_selection import train_test_split
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def ransac_regression_model(data, predictor_variables, stage_name, n_seeds=25):
    """
    Construir y entrenar un modelo de regresión RANSAC (robusto a outliers)
    para predecir el consumo de combustible basado en múltiples variables predictoras
    """
    print(f"\n=== MODELO DE REGRESIÓN RANSAC - STAGE {stage_name} ===")
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

    # ============ OPTIMIZACIÓN RANSAC ============
    print("Optimizando hiperparámetros RANSAC...")

    n_samples, n_features = X_train_scaled.shape

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
    best_ransac_model = None
    best_params = None
    best_optimization_metrics = {}

    total_tests = len(configurations) * n_seeds
    current_test = 0

    for config_idx, config in enumerate(configurations):
        print(f"Probando configuración {config_idx + 1}/{len(configurations)}")

        for seed in range(n_seeds):
            current_test += 1
            try:
                params = config.copy()
                params["random_state"] = seed

                # Entrenar modelo RANSAC
                ransac = linear_model.RANSACRegressor(**params)
                ransac.fit(X_train_scaled, y_train)

                # Calcular métricas en datos de entrenamiento
                y_train_pred = ransac.predict(X_train_scaled)
                r2_train = r2_score(y_train, y_train_pred)
                mae_train = mean_absolute_error(y_train, y_train_pred)

                # Métricas específicas de RANSAC
                inlier_mask = ransac.inlier_mask_
                n_inliers = np.sum(inlier_mask)
                inlier_ratio = n_inliers / len(y_train)

                # R² en inliers
                if n_inliers > n_features + 1:
                    r2_inliers = r2_score(
                        y_train[inlier_mask], y_train_pred[inlier_mask]
                    )
                else:
                    r2_inliers = -np.inf

                # Score compuesto para optimización
                composite_score = (
                    0.6 * max(0, r2_train)
                    + 0.2 * max(0, r2_inliers)
                    + 0.15 * inlier_ratio
                    + 0.05 * max(0, 1 - mae_train / (np.max(y_train) - np.min(y_train)))
                )

                if composite_score > best_score and r2_train > 0:
                    best_score = composite_score
                    best_ransac_model = ransac
                    best_params = params.copy()
                    best_optimization_metrics = {
                        "r2_train": r2_train,
                        "r2_inliers": r2_inliers,
                        "mae_train": mae_train,
                        "n_inliers": n_inliers,
                        "inlier_ratio": inlier_ratio,
                        "composite_score": composite_score,
                        "n_trials": ransac.n_trials_,
                    }

                if current_test % 10 == 0:
                    progress = (current_test / total_tests) * 100
                    best_r2 = best_optimization_metrics.get("r2_train", 0)
                    print(f"Progreso: {progress:.1f}% - Mejor R²: {best_r2:.4f}")

            except Exception as e:
                print(f"Error en iteración {current_test}: {str(e)}")
                continue

    if best_ransac_model is None:
        raise ValueError("No se pudo optimizar el modelo RANSAC. Verifique los datos.")

    print(f"\n✓ Optimización completada. Mejor configuración encontrada:")
    print(f"  - R² entrenamiento: {best_optimization_metrics['r2_train']:.4f}")
    print(f"  - Ratio de inliers: {best_optimization_metrics['inlier_ratio']:.4f}")
    print(f"  - N° inliers: {best_optimization_metrics['n_inliers']}")

    # Realizar predicciones finales en conjunto de prueba
    y_train_pred = best_ransac_model.predict(X_train_scaled)
    y_test_pred = best_ransac_model.predict(X_test_scaled)

    # Calcular métricas de evaluación finales
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Métricas RANSAC en datos de entrenamiento
    inlier_mask_final = best_ransac_model.inlier_mask_
    n_inliers_final = np.sum(inlier_mask_final)
    inlier_ratio_final = n_inliers_final / len(y_train)

    print(f"\n=== MÉTRICAS FINALES DEL MODELO RANSAC ===")
    print(f"R² Entrenamiento: {train_r2:.4f}")
    print(f"R² Prueba:        {test_r2:.4f}")
    print(f"MAE Prueba:       {test_mae:.4f} litros")
    print(f"RMSE Prueba:      {test_rmse:.4f} litros")
    print(
        f"Inliers:          {n_inliers_final}/{len(y_train)} ({inlier_ratio_final:.2%})"
    )
    print(f"Iteraciones RANSAC: {best_ransac_model.n_trials_}")

    # Mostrar información del modelo estimador base
    base_estimator = best_ransac_model.estimator_
    print(f"\nEcuación del modelo RANSAC (variables estandarizadas):")
    print(f"FuelConsumed = {base_estimator.intercept_:.4f}", end="")
    for i, coef in enumerate(base_estimator.coef_[: len(predictor_variables)]):
        print(f" + ({coef:.4f} × {predictor_variables[i]})", end="")
    print(f" + términos_destination")

    # Retornar diccionario con todos los componentes del modelo
    return {
        "ransac_regression_model": best_ransac_model,  # Modelo RANSAC entrenado
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
        "inlier_mask": inlier_mask_final,  # Máscara de inliers
        "n_inliers": n_inliers_final,  # Número de inliers
        "inlier_ratio": inlier_ratio_final,  # Proporción de inliers
        "ransac_trials": best_ransac_model.n_trials_,  # Número de iteraciones RANSAC
        "best_ransac_params": best_params,  # Mejores hiperparámetros encontrados
        "optimization_metrics": best_optimization_metrics,  # Métricas del proceso de optimización
    }
