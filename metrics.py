import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from config import GRAPH_DIR


# ======================================
# RMSE
# ======================================

def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


# ======================================
# MAE
# ======================================

def mae(y_true, y_pred):

    return mean_absolute_error(
        y_true,
        y_pred
    )


# ======================================
# MSE
# ======================================

def mse(y_true, y_pred):

    return mean_squared_error(
        y_true,
        y_pred
    )


# ======================================
# MAPE
# ======================================

def mape(y_true, y_pred):

    y_true = np.array(y_true)

    y_pred = np.array(y_pred)

    return np.mean(

        np.abs(

            (y_true - y_pred)

            / y_true

        )

    ) * 100


# ======================================
# R2 Score
# ======================================

def r2(y_true, y_pred):

    return r2_score(
        y_true,
        y_pred
    )


# ======================================
# Print Metrics
# ======================================

def evaluate(y_true, y_pred):

    print("=" * 50)

    print(f"RMSE : {rmse(y_true,y_pred):.4f}")

    print(f"MAE  : {mae(y_true,y_pred):.4f}")

    print(f"MSE  : {mse(y_true,y_pred):.4f}")

    print(f"MAPE : {mape(y_true,y_pred):.2f}%")

    print(f"R²   : {r2(y_true,y_pred):.4f}")

    print("=" * 50)


# ======================================
# Actual vs Predicted Graph
# ======================================

def save_prediction_plot(

    actual,

    predicted

):

    plt.figure(figsize=(12,6))

    plt.plot(

        actual,

        label="Actual Price"

    )

    plt.plot(

        predicted,

        label="Predicted Price"

    )

    plt.xlabel("Days")

    plt.ylabel("Price")

    plt.title("Actual vs Predicted")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "prediction.png"

        )

    )

    plt.close()


# ======================================
# Loss Curve
# ======================================

def save_loss_plot(

    train_loss,

    val_loss

):

    plt.figure(figsize=(10,5))

    plt.plot(

        train_loss,

        label="Train Loss"

    )

    plt.plot(

        val_loss,

        label="Validation Loss"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            GRAPH_DIR,

            "loss_curve.png"

        )

    )

    plt.close()