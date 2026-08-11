import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from config import SEQUENCE_LENGTH, BATCH_SIZE


class DataPreprocessor:

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def normalize(self, df):

        # Remove invalid rows
        df = df[df["Date"].notna()].copy()

        # Convert Close column to numeric
        df["Close"] = df["Close"].astype(float)

        data = df["Close"].values.reshape(-1, 1)

        scaled = self.scaler.fit_transform(data)

        return scaled

    def create_sequences(self, data):

        X = []
        y = []

        for i in range(SEQUENCE_LENGTH, len(data)):
            X.append(data[i - SEQUENCE_LENGTH:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def split(self, X, y, test_size=0.2):

        return train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=False
        )

    def loaders(self, X_train, X_test, y_train, y_test):

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        return train_loader, test_loader