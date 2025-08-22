from pathlib import Path

# --- File System Paths ---
UPLOAD_FOLDER = Path("uploads/")
MODELS_FOLDER = Path("models/")

# --- Model & Data Configuration ---
DATASET_URLS = {
    "pizza_steak_sushi_10_percent": "https://github.com/GoJo-Rika/PyTorch-FoodVision-Mini/raw/main/data/pizza_steak_sushi.zip",
    "pizza_steak_sushi_20_percent": "https://github.com/GoJo-Rika/PyTorch-FoodVision-Mini/raw/main/data/pizza_steak_sushi_20_percent.zip",
}
CLASS_NAMES = ["pizza", "steak", "sushi"]
DEFAULT_MODEL = "effnetb0"

# --- File Upload Validation ---
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
