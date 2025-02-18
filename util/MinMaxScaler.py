from typing import List

# def min_max_scaler(data: List[List[float]]) -> List[List[float]]:
#     if not data or not data[0]:  # Handle empty input
#         return []
#
#     num_cols = len(data[0])
#
#     # Transpose to get columns
#     cols = list(zip(*data))
#
#     # Compute min and max for each column
#     mins = [min(col) for col in cols]
#     maxs = [max(col) for col in cols]
#
#     # Apply min-max scaling
#     scaled_data = [
#         [
#             (val - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0
#             for j, val in enumerate(row)
#         ]
#         for row in data
#     ]
#
#     return scaled_data

class MinMaxScaler:
    def __init__(self):
        self.mins = []
        self.maxs = []

    def transform(self, data: List[List[float]]) -> List[List[float]]:
        # num_cols = len(data[0])

        # Transpose to get columns
        cols = list(zip(*data))

        # Compute min and max for each column
        mins = [min(col) for col in cols]
        maxs = [max(col) for col in cols]

        # Apply min-max scaling
        scaled_data = [
            [
                (val - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0
                for j, val in enumerate(row)
            ]
            for row in data
        ]

        self.mins = mins
        self.maxs = maxs

        return scaled_data

    def inverse_transform(self, data: List[List[float]]) -> List[List[float]]:
        unscaled = []
        for row in data:
            unscaled_row = []
            for j, val in enumerate(row):
                # Reconstruct original value
                original = val * (self.maxs[j] - self.mins[j]) + self.mins[j]
                unscaled_row.append(original)
            unscaled.append(unscaled_row)

        return unscaled