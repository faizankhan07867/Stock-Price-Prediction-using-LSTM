import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("dataset/stock_data.csv")

close_prices = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(close_prices)

joblib.dump(scaler, "model/scaler.pkl")

np.save("model/scaled_data.npy", scaled_data)

print("Preprocessing Completed Successfully")