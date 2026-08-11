import os
import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from config import *

from models.lstm import create_model
from utils.dataset import StockDataset
from utils.preprocessing import DataPreprocessor
from utils.logger import *
from utils.metrics import *

# ======================================
# Download / Load Dataset
# ======================================

stock = StockDataset(DEFAULT_STOCK)

df = stock.load()

# ======================================
# Preprocessing
# ======================================

processor = DataPreprocessor()

scaled = processor.normalize(df)

X, y = processor.create_sequences(scaled)

X_train, X_test, y_train, y_test = processor.split(
    X,
    y
)

train_loader, test_loader = processor.loaders(
    X_train,
    X_test,
    y_train,
    y_test
)

# Save scaler

joblib.dump(
    processor.scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

# ======================================
# Model
# ======================================

model = create_model()

criterion = nn.MSELoss()

optimizer = optim.Adam(

    model.parameters(),

    lr=LEARNING_RATE

)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=5

)

train_losses = []

val_losses = []

best_loss = float("inf")

log_info("Training Started")

# ======================================
# Training Function
# ======================================

def train_one_epoch(epoch):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for sequences, labels in progress:

        sequences = sequences.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(sequences)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}"
        )

    epoch_loss = running_loss / len(train_loader)

    train_losses.append(epoch_loss)

    return epoch_loss


# ======================================
# Validation Function
# ======================================

def validate():

    model.eval()

    running_loss = 0.0

    predictions = []

    actuals = []

    with torch.no_grad():

        for sequences, labels in test_loader:

            sequences = sequences.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(sequences)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predictions.extend(
                outputs.cpu().numpy()
            )

            actuals.extend(
                labels.cpu().numpy()
            )

    val_loss = running_loss / len(test_loader)

    val_losses.append(val_loss)

    scheduler.step(val_loss)

    return (

        val_loss,

        actuals,

        predictions

    )
    
# ======================================
# Main Training Loop
# ======================================

if __name__ == "__main__":

    print("=" * 60)
    print("📈 Stock Price Prediction using LSTM")
    print("=" * 60)

    start_time = time.time()

    best_predictions = None
    best_actuals = None

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(epoch)

        val_loss, actuals, predictions = validate()

        log_epoch(
            epoch,
            train_loss,
            val_loss
        )

        print(f"\nEpoch [{epoch}/{EPOCHS}]")
        print(f"Train Loss      : {train_loss:.6f}")
        print(f"Validation Loss : {val_loss:.6f}")

        # ==========================
        # Save Best Model
        # ==========================

        if val_loss < best_loss:

            best_loss = val_loss

            best_predictions = predictions

            best_actuals = actuals

            torch.save(

                model.state_dict(),

                MODEL_PATH

            )

            log_info("Best Model Saved")

    # ==========================
    # Inverse Transform
    # ==========================

    actuals = processor.scaler.inverse_transform(
        best_actuals
    )

    predictions = processor.scaler.inverse_transform(
        best_predictions
    )

    # ==========================
    # Evaluation
    # ==========================

    evaluate(

        actuals,

        predictions

    )

    save_prediction_plot(

        actuals,

        predictions

    )

    save_loss_plot(

        train_losses,

        val_losses

    )

    # ==========================
    # Training Time
    # ==========================

    end_time = time.time()

    minutes = (

        end_time - start_time

    ) / 60

    print("=" * 60)

    print("Training Completed Successfully")

    print(f"Best Validation Loss : {best_loss:.6f}")

    print(f"Training Time : {minutes:.2f} Minutes")

    print(f"Model Saved : {MODEL_PATH}")

    print("=" * 60)

    log_info("Training Finished")