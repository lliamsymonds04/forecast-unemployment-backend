import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dateutil.relativedelta import relativedelta

from DataLoader import DataFrameLike
from util.MinMaxScaler import MinMaxScaler
from util.DateUtils import get_data_in_range, str_to_datetime, get_date_ranges, find_date_index


class UnemploymentForecastModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers=1):
        super(UnemploymentForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # Use the last time step's output
        return out


def create_sequences(data: torch.Tensor, seq_length: int):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :]  # Use all features as input
        y = data[i+seq_length, 0]    # Predict Unemployment (first column)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


class PredictionModel:
    def __init__(self, df: DataFrameLike, training_start_date: str, training_end_date: str, sequence_length: int, num_epochs: int):
        self.df = df
        self.training_start_date = training_start_date
        self.training_end_date = training_end_date
        self.sequence_length = sequence_length
        self.num_epochs = num_epochs

        training_df = get_data_in_range(self.df, training_start_date, training_end_date)
        test_df = get_data_in_range(self.df, training_end_date)

        self.training_df = training_df
        self.test_df = test_df
        self.scaler = MinMaxScaler()

        test_scaled = self.scaler.transform(test_df)
        training_scaled = self.scaler.transform(training_df)

        training_data = torch.tensor(training_scaled, dtype=torch.float32)
        test_data = torch.tensor(test_scaled, dtype=torch.float32)

        X_train, y_train = create_sequences(training_data, sequence_length)
        X_test, y_test = create_sequences(test_data, sequence_length)

        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Initialize the model
        model = UnemploymentForecastModel(input_size=len(df["columns"]), hidden_size=20, output_size=1)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0  # Initialize loss for the epoch
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()  # Accumulate loss for the epoch

        self.model = model
        self.test_data = test_data
        self.X_test = X_test
        self.y_test = y_test


    def predict(self, months: int, interest_rate: float, inflation_rate: float):
        new_input = self.training_df[-self.sequence_length:]

        inflation_multi = math.pow((100+inflation_rate)/100, 1/12)

        forecasted_months = []
        predictions = []

        start_date = str_to_datetime(self.training_end_date)

        for _ in range(months):
            new_input_scaled = self.scaler.transform(new_input)

            # Convert to PyTorch tensor
            new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32).unsqueeze(0)

            # Predict using the model
            self.model.eval()
            with torch.no_grad():
                prediction: torch.Tensor = self.model(new_input_tensor)

            v: float = prediction.item()
            original = self.scaler.inverse_transform([[v,0.0,0.0]])[0][0]

            predictions.append(original)
            new_input.pop(0)
            new_input.append([original, new_input[-1][1] * inflation_multi, interest_rate])

            start_date += relativedelta(months=1)
            forecasted_months.append(start_date.strftime("%Y-%m-%d"))

        return {
            "index": forecasted_months,
            "data": predictions,
        }


    def evaluate_model(self, evaluation_range: int):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions: np.ndarray = self.model(self.X_test).squeeze().numpy()

        scaled: list[float] = predictions.tolist()

        useless_array = [0.0] * len(scaled)

        zipped = list(zip(scaled, useless_array, useless_array))
        zipped_lists = [list(item) for item in zipped]
        unscaled = self.scaler.inverse_transform(zipped_lists)

        expected_df = self.test_df[:-self.sequence_length]
        expected = list(zip(*expected_df))[0][:evaluation_range]
        observed = list(zip(*unscaled))[0][:evaluation_range]

        test_date_index = find_date_index(self.df, str_to_datetime(self.training_end_date)) + 1
        dates = self.df["index"][test_date_index:(test_date_index+evaluation_range)]


        return {
            "index": dates,
            "expected": expected,
            "predictions": observed,
        }