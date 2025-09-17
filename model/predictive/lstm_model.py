import torch
import torch.nn as nn
import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")


class SimpleLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=5, num_layers=2):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class FuelLevelPredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

    def load_data(self, file_path):
        print("Cargando datos...")
        self.df = pl.read_csv(file_path)
        print(f"Datos cargados: {len(self.df)} registros")
        return self.df

    def prepare_data(self):
        # Usar solo FuelLevelLiters
        fuel_data = self.df["FuelLevelLiters"].to_numpy()

        # Crear secuencias
        X, y = [], []
        for i in range(self.sequence_length, len(fuel_data)):
            X.append(fuel_data[i - self.sequence_length : i])
            y.append(fuel_data[i])

        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)

        # Dividir en entrenamiento y prueba (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Convertir a tensores
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        # Crear DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Mover datos de prueba a GPU
        self.X_test = X_test_tensor.to(self.device)
        self.y_test = y_test_tensor.to(self.device)

        return len(X_train), len(X_test)

    def build_model(self):
        self.model = SimpleLSTMModel().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=10):
        train_size, test_size = self.prepare_data()
        self.build_model()

        # Entrenamiento
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Evaluación
        self.model.eval()
        with torch.no_grad():
            # Evaluar en lotes para ahorrar memoria
            test_preds = []
            batch_size = 32
            for i in range(0, len(self.X_test), batch_size):
                batch = self.X_test[i : i + batch_size]
                pred_batch = self.model(batch).squeeze()
                test_preds.append(pred_batch.cpu().numpy())

            test_preds = np.concatenate(test_preds)
            test_actual = self.y_test.cpu().numpy()

            mse = mean_squared_error(test_actual, test_preds)
            mae = mean_absolute_error(test_actual, test_preds)
            rmse = np.sqrt(mse)

            print(f"\nMétricas en conjunto de prueba:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            return mse, mae, rmse

    def predict_all_data(self):
        print("Generando predicciones para todos los datos...")

        # Preparar datos completos
        fuel_data = self.df["FuelLevelLiters"].to_numpy()

        predictions = []

        # Para los primeros registros, usar valores reales
        for i in range(self.sequence_length):
            predictions.append(self.df["FuelLevelLiters"][i])

        # Predecir el resto en lotes para ahorrar memoria
        # Nota: Esto puede ser lento porque predecimos uno por uno, pero es necesario para la secuencia
        for i in range(self.sequence_length, len(self.df)):
            seq = fuel_data[i - self.sequence_length : i]
            seq_tensor = (
                torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1).to(self.device)
            )

            self.model.eval()
            with torch.no_grad():
                pred = self.model(seq_tensor).cpu().numpy()[0, 0]
                predictions.append(pred)

            if i % 10000 == 0:
                print(f"Procesados {i} de {len(self.df)} registros")

        # Crear copia del DataFrame original y reemplazar solo FuelLevelLiters
        df_pred = self.df.clone()
        df_pred = df_pred.with_columns(pl.Series("FuelLevelLiters", predictions))

        # Guardar archivo
        df_pred.write_csv("prediction_lstm.csv")
        print("Archivo 'prediction_lstm.csv' guardado exitosamente!")

        return df_pred


# Ejecutar el modelo
if __name__ == "__main__":
    # Crear y entrenar el modelo
    predictor = FuelLevelPredictor(sequence_length=10)
    df = predictor.load_data("unified_data_T-210.csv")

    print("\nEntrenando modelo...")
    metrics = predictor.train(epochs=20)

    print("\nGenerando predicciones completas...")
    df_predictions = predictor.predict_all_data()
    print(f"Predicciones completadas para {len(df_predictions)} registros")
