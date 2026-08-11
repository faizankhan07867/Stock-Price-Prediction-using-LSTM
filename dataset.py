import os
import yfinance as yf
import pandas as pd

from config import *


class StockDataset:

    def __init__(self, symbol=DEFAULT_STOCK):

        self.symbol = symbol

        self.csv_path = os.path.join(
            DATASET_DIR,
            f"{symbol}.csv"
        )

    # ======================================
    # Download Dataset
    # ======================================

    def download(self):

        print(f"\nDownloading {self.symbol}...")

        df = yf.download(

            self.symbol,

            start=START_DATE,

            end=END_DATE,

            progress=False

        )

        if df.empty:

            raise Exception(
                "No data downloaded."
            )

        df.reset_index(inplace=True)

        df.to_csv(

            self.csv_path,

            index=False

        )

        print("Dataset Saved :")

        print(self.csv_path)

        return df

    # ======================================
    # Load Dataset
    # ======================================

    def load(self):

        if os.path.exists(self.csv_path):

            print("Loading Existing Dataset...")

            return pd.read_csv(
                self.csv_path
            )

        return self.download()

    # ======================================
    # Information
    # ======================================

    def info(self):

        df = self.load()

        print("=" * 50)

        print(df.head())

        print()

        print(df.info())

        print()

        print(df.describe())

        print("=" * 50)

        return df


# ======================================
# Test
# ======================================

if __name__ == "__main__":

    stock = StockDataset("AAPL")

    data = stock.info()

    print(data.tail())