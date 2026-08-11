import os
import torch

# ======================================
# Base Directory
# ======================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================
# Dataset
# ======================================

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# ======================================
# Outputs
# ======================================

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_model.pth"
)

GRAPH_DIR = os.path.join(
    OUTPUT_DIR,
    "graphs"
)

LOG_DIR = os.path.join(
    OUTPUT_DIR,
    "logs"
)

PREDICTION_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)

# ======================================
# Stock Settings
# ======================================

DEFAULT_STOCK = "AAPL"

START_DATE = "2018-01-01"

END_DATE = None

# ======================================
# Sequence
# ======================================

SEQUENCE_LENGTH = 60

# ======================================
# Training
# ======================================

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 0.001

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.2

# ======================================
# Device
# ======================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ======================================
# Random Seed
# ======================================

SEED = 42

# ======================================
# Create Directories
# ======================================

os.makedirs(DATASET_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(GRAPH_DIR, exist_ok=True)

os.makedirs(LOG_DIR, exist_ok=True)

os.makedirs(PREDICTION_DIR, exist_ok=True)

# ======================================
# Print Config
# ======================================

print("=" * 60)

print("Stock Price Prediction using LSTM")

print("=" * 60)

print("Device :", DEVICE)

print("Epochs :", EPOCHS)

print("Batch Size :", BATCH_SIZE)

print("Learning Rate :", LEARNING_RATE)

print("Sequence Length :", SEQUENCE_LENGTH)

print("Hidden Size :", HIDDEN_SIZE)

print("=" * 60)