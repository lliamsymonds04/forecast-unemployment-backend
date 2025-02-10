import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


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
    def __init__(self, df: pd.DataFrame, training_start_date: str, training_end_date: str, sequence_length: int, num_epochs: int):
        self.df = df
        self.training_start_date = training_start_date
        self.training_end_date = training_end_date
        self.sequence_length = sequence_length
        self.num_epochs = num_epochs

        training_df = df.loc[(df.index >= training_start_date) & (df.index < training_end_date)]
        test_df = df.loc[df.index >= training_end_date]
        scaler = MinMaxScaler()

        self.training_df = training_df
        self.test_df = test_df
        self.scaler = scaler

        training_scaled = scaler.fit_transform(training_df)
        test_scaled = scaler.transform(test_df)

        training_data = torch.tensor(training_scaled, dtype=torch.float32)
        test_data = torch.tensor(test_scaled, dtype=torch.float32)

        X_train, y_train = create_sequences(training_data, sequence_length)
        X_test, y_test = create_sequences(test_data, sequence_length)

        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize the model
        model = UnemploymentForecastModel(input_size=training_df.shape[1], hidden_size=50, output_size=1)

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

    def predict(self, months: int, interest_rate: float, inflation_rate: float):
        # Get the last unemployment sequence
        last_unemployment = self.training_df["Unemployment"].values[-self.sequence_length:]

        # Set fixed values for interest rate and inflation
        inflation_mult = math.pow(inflation_rate, 1/12)
        current_inflation = self.training_df["Inflation"].values[-1] * inflation_mult

        # Store predictions
        predictions = []

        # Loop for multiple forecast steps
        for _ in range(months):
            # Create a new input sequence
            new_input = np.column_stack((
                last_unemployment,  # Unemployment (previous step)
                np.full(self.sequence_length, current_inflation),  # Inflation
                np.full(self.sequence_length, interest_rate)  # Interest Rate
            ))

            # Convert to DataFrame and normalize
            new_input_df = pd.DataFrame(new_input, columns=self.training_df.columns)
            new_input_scaled = self.scaler.transform(new_input_df)

            # Convert to PyTorch tensor
            new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32).unsqueeze(0)

            # Predict using the model
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(new_input_tensor)

            # Inverse transform the prediction
            prediction_original = self.scaler.inverse_transform(
                np.concatenate([prediction.numpy(), np.zeros((1, self.training_df.shape[1] - 1))], axis=1)
            )[0, 0]

            # Append the prediction
            predictions.append(float(prediction_original))

            # Update last unemployment values by shifting and adding new prediction
            last_unemployment = np.roll(last_unemployment, -1)
            last_unemployment[-1] = prediction_original

            #update the inflation trend
            current_inflation *= inflation_mult

        return predictions