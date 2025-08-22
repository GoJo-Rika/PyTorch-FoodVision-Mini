import base64
import io
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

# Import from the core ML (food_vision) package
from food_vision import model_builder
from food_vision import predict as predictor

# Import from our new shared modules (webapp)
from webapp import config, state


# --- File Validation Function ---
def is_allowed_file(filename: str) -> bool:
    """Checks if a filename has an allowed image extension."""
    if not filename or "." not in filename:
        return False
    # Using pathlib to robustly get the extension and convert to lowercase
    ext = Path(filename).suffix.lower()
    return ext in config.ALLOWED_EXTENSIONS


def get_pytorch_device():
    """Returns the appropriate PyTorch device string ('cuda', 'mps', or 'cpu')."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_info():
    """Gets a user-friendly string for the available compute device."""
    device = get_pytorch_device()
    if device == "cuda":
        return "NVIDIA GPU (CUDA)"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def get_available_models():
    """Scans the models directory and returns a list of .pth files."""
    if not config.MODELS_FOLDER.exists():
        return []
    return sorted([f.name for f in config.MODELS_FOLDER.glob("*.pth")], reverse=True)


def check_training_status():
    """Checks and updates the global training status."""
    just_finished = False
    message = None
    if state.training_process:
        return_code = state.training_process.poll()
        if return_code is not None:  # Process has finished
            if state.training_log["status"] == "Running":
                just_finished = True
                model_filename = state.training_log.get("model_filename", "The last model")
                if return_code == 0:
                    message = f"Training for '{model_filename}' completed successfully!"
                    state.training_log["details"] = (f"Last run for '{model_filename}' completed.")
                else:
                    message = f"Training for '{model_filename}' failed. Check terminal."
                    state.training_log["details"] = (f"Last run for '{model_filename}' failed.")

            state.training_log["status"] = "Idle"
            state.training_process = None
    return just_finished, message


def start_training_process(model: str, data_name: str, epochs: int, lr: float, batch_size: int):
    """Launches the training script in a subprocess and updates the state."""
    if state.training_process and state.training_process.poll() is None:
        return False, "A training process is already running."

    model_filename = f"{model}_{data_name}_{epochs}_epochs.pth"
    state.training_log["model_filename"] = model_filename

    command = [
        "python", "train.py",
            "--model", model,
            "--data_name", data_name,
            "--data_url", config.DATASET_URLS.get(data_name),
            "--epochs", str(epochs),
            "--learning_rate", str(lr),
            "--batch_size", str(batch_size),
    ]

    print(f"Running command: {' '.join(command)}")
    state.training_process = subprocess.Popen(command)

    state.training_log["status"] = "Running"
    state.training_log["details"] = f"Training model: {model_filename}"
    return True, f"Started training for {model_filename}."


def cancel_current_training():
    """Cancels the currently active training process."""
    if state.training_process and state.training_process.poll() is None:
        model_filename = state.training_log.get("model_filename", "the model")
        state.training_process.terminate()
        state.training_process = None
        state.training_log["status"] = "Idle"
        state.training_log["details"] = (f"Training for '{model_filename}' was cancelled.")
        return True, "Training process has been cancelled."
    return False, "No training process is currently running."


def perform_prediction(model_path_str: str, image_path: str):
    """Loads a model and performs prediction on an image."""
    try:
        full_model_path = config.MODELS_FOLDER / model_path_str

        match = re.search(r"effnetb\d+", full_model_path.name)
        model_name = match.group(0) if match else config.DEFAULT_MODEL

        device = torch.device(get_pytorch_device())

        model = model_builder.EfficientNet(model_name=model_name, num_classes=len(config.CLASS_NAMES))
        model.load_state_dict(torch.load(full_model_path, map_location=device))
        model.to(device)

        img, pred_title = predictor.pred_and_plot_image(
            model=model,
            image_path=str(image_path),
            class_names=config.CLASS_NAMES,
            device=device,
            return_fig=True,
        )

        buf = io.BytesIO()
        plt.imshow(img)
        plt.title(pred_title)
        plt.axis(False)
        plt.savefig(buf, format="png")
        plt.close()

        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return image_base64, None

    except Exception as e:
        print(f"Prediction error: {e}")
        return None, f"An error occurred: {e}"
