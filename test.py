import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
print(mlflow.get_tracking_uri())
# Debe imprimir: http://localhost:5000

# Crear experimento de prueba
mlflow.set_experiment("test_experiment")
with mlflow.start_run():
    mlflow.log_param("test", "value")
    print("Prueba exitosa!")
