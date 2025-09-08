# fuel_lstm_pytorch.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=5, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class FuelLevelPredictor:
    def __init__(self, sequence_length=60, test_size=0.2):
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_and_preprocess_data(self, filepath):
        self.data = pd.read_csv(filepath)
        self.data["TimeStamp"] = pd.to_datetime(self.data["TimeStamp"])
        self.data = self.data.sort_values("TimeStamp").reset_index(drop=True)
        self.data = self.data[["TimeStamp", "FuelLevelLiters"]].dropna()
        return self.data

    def create_sequences(self, data):
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def prepare_data(self):
        fuel_values = self.data["FuelLevelLiters"].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(fuel_values)
        X, y = self.create_sequences(scaled_data)
        train_size = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=32, shuffle=True
        )
        self.X_test, self.y_test = X_test.to(self.device), y_test.to(self.device)
        return X_train.shape, X_test.shape

    def build_model(self):
        self.model = LSTMModel().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_model(self, epochs=5):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def make_predictions(self):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.X_test).cpu().numpy()
            actual = self.y_test.cpu().numpy()
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(actual)
        mse = mean_squared_error(actual, predictions)
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mse)
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        return predictions, actual

    def plot_results(self, predictions, actual):
        test_timestamps = (
            self.data["TimeStamp"].iloc[-len(actual) :].reset_index(drop=True)
        )
        plt.figure(figsize=(12, 6))
        plt.plot(test_timestamps, actual, label="Actual")
        plt.plot(test_timestamps, predictions, label="Predicted")
        plt.xlabel("Time")
        plt.ylabel("Fuel Level (Liters)")
        plt.legend()
        plt.title("Fuel Level Prediction")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    predictor: FuelLevelPredictor = FuelLevelPredictor(sequence_length=60)
    current_file: Path = Path(__file__).resolve().parent.parent.parent
    data_path: Path = current_file / "frontend/web/app/output/T-210_sensor.csv"
    predictor.load_and_preprocess_data(data_path)
    predictor.prepare_data()
    predictor.build_model()
    predictor.train_model(epochs=5)
    predictions, actual = predictor.make_predictions()
    predictor.plot_results(predictions, actual)
