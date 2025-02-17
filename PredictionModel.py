import math
from flask import Response
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import io

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

# def convert_plt_to_img():
#     img = io.BytesIO()
#     plt.savefig(img, format='png', bbox_inches='tight')
#     plt.close()
#     img.seek(0)
#
#     return Response(img.getvalue(), mimetype='image/png')

class PredictionModel:
    def __init__(self, df: pd.DataFrame, training_start_date: str, training_end_date: str, sequence_length: int, num_epochs: int):
        self.df = df
        self.training_start_date = training_start_date
        self.training_end_date = training_end_date
        self.sequence_length = sequence_length
        self.num_epochs = num_epochs

        training_df: pd.DataFrame = df.loc[(df.index >= training_start_date) & (df.index < training_end_date)]
        test_df: pd.DataFrame = df.loc[df.index >= training_end_date]
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
        self.test_data = test_data
        self.X_test = X_test
        self.y_test = y_test

    def predict(self, months: int, interest_rate: float, inflation_rate: float):
        # Get the last unemployment sequence
        last_unemployment = self.training_df["Unemployment"].values[-self.sequence_length:]

        # Set fixed values for interest rate and inflation
        inflation_multi = math.pow(inflation_rate/100, 1/12)
        current_inflation = self.training_df["Inflation"].values[-1] * inflation_multi

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
            current_inflation *= inflation_multi

        return predictions


    def graph_predictions(self, predictions: list[float]):
        data: pd.DataFrame = self.training_df.copy()

        last_date = data.index.max()
        last_unemployment = data.iloc[-1, 0]
        future_dates = pd.date_range(start=last_date, periods=len(predictions) + 2, freq='MS')[1:]

        baseline_df = pd.DataFrame({'Unemployment_Predicted': [last_unemployment]}, index=[future_dates[0]])
        pred_df = pd.DataFrame({'Unemployment_Predicted': predictions}, index=future_dates[1:])
        pred_df = pd.concat([baseline_df, pred_df])
        data = pd.concat([data, pred_df])

        # plt.figure(figsize=(6, 4))
        # sns.lineplot(x=data.index, y="Unemployment", data=data)
        # sns.lineplot(x=data.index, y="Unemployment_Predicted", data=data, label="Predicted", linestyle="dashed")
        #
        # return convert_plt_to_img()
        return data


    def evaluate_model(self):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predictions = self.model(self.X_test).squeeze().numpy()

        # Convert back to original scale
        y_test_original = self.scaler.inverse_transform(
            np.concatenate([self.y_test.numpy().reshape(-1, 1), np.zeros((len(self.y_test), self.df.shape[1] - 1))],
                           axis=1))[:, 0]
        y_pred_original = self.scaler.inverse_transform(
            np.concatenate([predictions.reshape(-1, 1), np.zeros((len(predictions), self.df.shape[1] - 1))], axis=1))[:,
                          0]

        # Compute evaluation metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)

        # print(f"Mean Squared Error (MSE): {mse:.4f}")
        # print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        test_dates = self.test_df.index[self.sequence_length:]
        df_plot = pd.DataFrame({
            "Date": test_dates,  # You can replace with actual date indices
            "Actual": y_test_original,
            "Predicted": y_pred_original
        })

        return df_plot
        # # Set Seaborn style
        # sns.set_style("whitegrid")
        #
        # # Plot with Seaborn
        # plt.figure(figsize=(12, 6))
        # sns.lineplot(data=df_plot, x="Date", y="Actual", label="Actual", color="blue")
        # ax = sns.lineplot(data=df_plot, x="Date", y="Predicted", label="Predicted", color="red", linestyle="dashed")
        #
        # ax.text(0.7, 0.9, f"MSE: {round(mse, 3)}", transform=ax.transAxes)
        # ax.text(0.7, 0.85, f"RMSE: {round(rmse, 3)}", transform=ax.transAxes)
        #
        # plt.title("Predicted vs. Actual Unemployment Rate", fontsize=20)
        # plt.xticks(rotation=45)
        # plt.ylabel("Unemployment Rate %")
        # plt.tight_layout()
        #
        # return convert_plt_to_img()
