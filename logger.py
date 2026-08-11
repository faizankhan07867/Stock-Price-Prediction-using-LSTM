import os
import logging
from datetime import datetime

from config import LOG_DIR

# ======================================
# Create Log Directory
# ======================================

os.makedirs(LOG_DIR, exist_ok=True)

# ======================================
# Log File
# ======================================

LOG_FILE = os.path.join(
    LOG_DIR,
    f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

# ======================================
# Logger Configuration
# ======================================

logger = logging.getLogger("StockPrediction")

logger.setLevel(logging.INFO)

formatter = logging.Formatter(

    "%(asctime)s | %(levelname)s | %(message)s",

    datefmt="%Y-%m-%d %H:%M:%S"

)

# ======================================
# File Handler
# ======================================

file_handler = logging.FileHandler(LOG_FILE)

file_handler.setFormatter(formatter)

# ======================================
# Console Handler
# ======================================

console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)

if not logger.handlers:

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

# ======================================
# Functions
# ======================================

def log_info(message):

    logger.info(message)

def log_warning(message):

    logger.warning(message)

def log_error(message):

    logger.error(message)

def log_epoch(

    epoch,

    train_loss,

    val_loss

):

    logger.info(

        f"Epoch [{epoch}] | "

        f"Train Loss : {train_loss:.6f} | "

        f"Validation Loss : {val_loss:.6f}"

    )