import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_loader import convert_tsf_to_dataframe
import pandas as pd
import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class TimeModel(nn.Module):
    def __init__(self, seq_len, dim, depth, output_len):
        super(TimeModel, self).__init__()
        self.input_proj = nn.Linear(seq_len, dim)
        self.time_blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ) for _ in range(depth)]
        )
        self.output_proj = nn.Linear(dim, output_len)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.time_blocks(x)
        x = self.output_proj(x)
        return x


class TimeSeriesPredictor:
    def __init__(self, input_length=96, output_length=96, embed_dim=64, num_blocks=4, hidden_dim=128, lr=0.001, epochs=10, batch_size=32):
        self.input_length = input_length
        self.output_length = output_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the model
        self.model = TimeModel(
            seq_len=input_length,
            dim=embed_dim,
            depth=num_blocks,
            output_len=output_length
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data):
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).squeeze()
        X, y = [], []
        for i in range(len(data_scaled) - self.input_length - self.output_length + 1):
            X.append(data_scaled[i: i + self.input_length])
            y.append(data_scaled[i + self.input_length: i + self.input_length + self.output_length])
        return np.array(X), np.array(y)

    def train(self, train_data):
        X, y = self.preprocess_data(train_data)
        train_dataset = TimeSeriesDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            # 使用 tqdm 显示每个 epoch 的进度条
            epoch_progress = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch", ncols=100)

            for batch_X, batch_y in epoch_progress:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # 更新进度条中的损失信息
                epoch_progress.set_postfix(loss=total_loss / (epoch_progress.n + 1))

            # 每个 epoch 结束后显示损失信息
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")

    def predict(self, input_data, forecast_steps):
        self.model.eval()
        predictions = []
        current_input = self.scaler.transform(input_data[-self.input_length:].reshape(-1, 1)).squeeze()

        with torch.no_grad():
            for _ in range(forecast_steps // self.output_length):
                input_tensor = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred = self.model(input_tensor).squeeze().cpu().numpy()
                predictions.extend(pred)
                current_input = np.concatenate([current_input[self.output_length:], pred])

        predictions = np.array(predictions[:forecast_steps])
        return self.scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()

    def plot_predictions(self, actual, predicted, title="Time Series Forecasting", save_path =None, display =True):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(actual)), actual, label="Actual Data", color="blue")
        plt.plot(range(len(actual), len(actual) + len(predicted)), predicted, label="Predicted Data", color="orange")
        plt.axvline(len(actual), color='red', linestyle='--', label='Prediction Start')
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.grid()
        if save_path:
            plt.savefig(save_path)
        if display:
            plt.show()

# Predict for the next 30 days (assuming each day has 96 steps)
def predict_series(series_values, forecast_steps= 96 * 30,  displayTitle = 'Time Series', save_path=None,display=True):
    df = pd.DataFrame({
        'Demand': pd.to_numeric(series_values, errors='coerce') 
    }) 
    # 设置日期为索引
    null_data = df['Demand'][df['Demand'].isnull()]
    df['Demand'].ffill( inplace=True)
    #df['Demand'].interpolate(method='linear', inplace=True)
    null_data2 = df['Demand'][df['Demand'].isnull()]
    time_series_data = df['Demand'].values

    # Initialize predictor
    predictor = TimeSeriesPredictor(input_length=96*7, output_length=96, epochs=100)

    # Train model
    train_data = time_series_data[-2000:]#[0:2000]
    predictor.train(train_data)

    # Predict for the next 30 days (assuming each day has 96 steps)
    predictions = predictor.predict(time_series_data, forecast_steps)

    # Visualize predictions
    predictor.plot_predictions(time_series_data[-2000:], predictions, title=f"{displayTitle} Prediction for Next 30 Days", save_path=save_path,display=display)

    return predictions

if __name__ == "__main__":
    # Generate synthetic time series data
    np.random.seed(42)
    #time_series_data = np.sin(np.linspace(0, 10 * np.pi, 1440)) + np.random.normal(0, 0.1, 1440)

    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("data/phase_1_data.tsf")
    start_timestamp= loaded_data['start_timestamp'][1]
    series_values= np.array(loaded_data['series_value'][0])
    #time_series_data=series_values[-1000:]

    predict_series(series_values=series_values,display=True)
