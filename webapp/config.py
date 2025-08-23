from pathlib import Path

# --- File System Paths ---
STATIC_FOLDER = Path("static/")
SAMPLES_FOLDER = STATIC_FOLDER / "samples"
UPLOAD_FOLDER = Path("uploads/")
MODELS_FOLDER = Path("models/")

# --- Model & Data Configuration ---
DATASET_URLS = {
    "pizza_steak_sushi_10_percent": "https://github.com/GoJo-Rika/PyTorch-FoodVision-Mini/raw/main/data/pizza_steak_sushi.zip",
    "pizza_steak_sushi_20_percent": "https://github.com/GoJo-Rika/PyTorch-FoodVision-Mini/raw/main/data/pizza_steak_sushi_20_percent.zip",
}

# --- Sample Image Configuration ---
SAMPLE_IMAGE_URLS = {
    "sample_pizza.jpg": "https://raw.githubusercontent.com/GoJo-Rika/datasets/refs/heads/main/PyTorch-FoodVision-Mini/pizza.jpeg",
    "sample_sushi.jpg": "https://raw.githubusercontent.com/GoJo-Rika/datasets/refs/heads/main/PyTorch-FoodVision-Mini/sushi.jpg",
    "sample_steak.jpg": "https://raw.githubusercontent.com/GoJo-Rika/datasets/refs/heads/main/PyTorch-FoodVision-Mini/steak.jpg",
}

CLASS_NAMES = ["pizza", "steak", "sushi"]
DEFAULT_MODEL = "effnetb0"

# --- File Upload Validation ---
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}
