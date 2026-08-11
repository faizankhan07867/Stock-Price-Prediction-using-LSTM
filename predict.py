import os
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from models.lstm import create_model
from utils.dataset import StockDataset

# ======================================
# Load Model
# ======================================

model = create_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# ======================================
# Load Scaler
# ======================================

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

# ======================================
# Prediction Function
# ======================================

def predict_stock(symbol=DEFAULT_STOCK, days=7):

    stock = StockDataset(symbol)

    df = stock.load()

    close = df["Close"].values.reshape(-1,1)

    scaled = scaler.transform(close)

    sequence = scaled[-SEQUENCE_LENGTH:]

    sequence = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    predictions = []

    for _ in range(days):

        with torch.no_grad():

            pred = model(sequence)

        predictions.append(pred.item())

        next_value = pred.cpu().numpy().reshape(1,1)

        sequence = sequence.cpu().numpy()

        sequence = np.append(
            sequence[:,1:,:],
            [[[next_value[0][0]]]],
            axis=1
        )

        sequence = torch.tensor(
            sequence,
            dtype=torch.float32
        ).to(DEVICE)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1,1)
    )

    print("="*60)

    print(f"Stock : {symbol}")

    print("="*60)

    for i, price in enumerate(predictions):

        print(

            f"Day {i+1} : ₹{price[0]:.2f}"

        )

    plt.figure(figsize=(10,5))

    plt.plot(

        range(1,days+1),

        predictions,

        marker="o"

    )

    plt.title(

        f"{symbol} Future Prediction"

    )

    plt.xlabel("Future Days")

    plt.ylabel("Predicted Price")

    plt.grid(True)

    plt.show()

    prediction_df = pd.DataFrame({

        "Day":[

            f"Day {i+1}"

            for i in range(days)

        ],

        "Predicted Price":[

            round(p[0],2)

            for p in predictions

        ]

    })

    prediction_df.to_csv(

        os.path.join(

            PREDICTION_DIR,

            f"{symbol}_prediction.csv"

        ),

        index=False

    )

    return prediction_df

# ======================================
# Main
# ======================================

if __name__=="__main__":

    symbol = input(

        "Enter Stock Symbol : "

    ).upper()

    predict_stock(symbol)