import base64
import io
import os
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


import matplotlib.pyplot as plt
import torch
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Import project's modules
from food_vision import model_builder
from food_vision import predict as predictor

# ---- Configuration ----
UPLOAD_FOLDER = Path("uploads/")
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER = Path("models/")
DATASET_URLS = {
    "pizza_steak_sushi_10_percent": "https://github.com/GoJo-Rika/PyTorch-FoodVision-Mini/raw/main/data/pizza_steak_sushi.zip",
    "pizza_steak_sushi_20_percent": "https://github.com/GoJo-Rika/PyTorch-FoodVision-Mini/raw/main/data/pizza_steak_sushi_20_percent.zip",
}

# ---- App State Management ----
training_process = None
training_log = {"status": "Idle", "details": "", "model_filename": ""}

# ---- Flask App Initialization ----
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---- Helper Functions ----
# Core function for getting the correct PyTorch device string
def get_pytorch_device():
    """Returns the appropriate PyTorch device string ('cuda', 'mps', or 'cpu')."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# This function uses the core function for display purposes
def get_device_info():
    """Gets the available compute device and returns a user-friendly formatted string."""
    device = get_pytorch_device()
    if device == "cuda":
        return "NVIDIA GPU (CUDA)"
    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"


def get_available_models():
    if not MODELS_FOLDER.exists(): 
        return []
    return sorted([f.name for f in MODELS_FOLDER.glob("*.pth")], reverse=True)


def check_training_status():
    global training_process, training_log
    if training_process:
        return_code = training_process.poll()
        if return_code is None:
            training_log["status"] = "Running"
        else:
            if training_log["status"] == "Running":
                model_filename = training_log.get("model_filename", "The last model")
                if return_code == 0:
                    flash(f"Training for '{model_filename}' completed successfully! It is now available for prediction.", "success")
                    training_log["details"] = f"Last run for '{model_filename}' completed successfully."
                else:
                    flash(f"Training for '{model_filename}' failed. Please check the terminal for errors.", "error")
                    training_log["details"] = f"Last run for '{model_filename}' failed. Check terminal for details."

            training_log["status"] = "Idle"
            training_process = None


# ---- Web App Routes ----
@app.route("/", methods=["GET"])
def home():
    check_training_status()
    available_models = get_available_models()
    device_info = get_device_info()
    return render_template(
        "index.html",
        models=available_models,
        training_status=training_log,
        device_info=device_info,
    )


@app.route("/status")
def status():
    check_training_status()
    return jsonify(training_log)


@app.route("/train", methods=["POST"])
def train():
    global training_process, training_log
    if training_process and training_process.poll() is None:
        flash("A training process is already running.", "error")
        return redirect(url_for("home"))

    model = request.form.get("model")
    data_name = request.form.get("data_name")
    epochs = request.form.get("epochs")
    lr = request.form.get("learning_rate")
    batch_size = request.form.get("batch_size")

    model_filename = f"{model}_{data_name}_{epochs}_epochs.pth"
    training_log["model_filename"] = model_filename

    command = [
        "python", "train.py", "--model", model, "--data_name", data_name,
        "--data_url", DATASET_URLS.get(data_name), "--epochs", str(epochs),
        "--learning_rate", str(lr), "--batch_size", str(batch_size),
    ]

    print(f"Running command: {' '.join(command)}")
    training_process = subprocess.Popen(command)

    training_log["status"] = "Running"
    training_log["details"] = f"Training model: {model_filename}"

    return redirect(url_for("home"))


@app.route("/cancel_training", methods=["POST"])
def cancel_training():
    global training_process, training_log
    if training_process and training_process.poll() is None:
        model_filename = training_log.get("model_filename", "the model")
        training_process.terminate()
        training_process = None
        training_log["status"] = "Idle"
        training_log["details"] = f"Training for '{model_filename}' was cancelled."
        flash("Training process has been cancelled.", "success")
    else:
        flash("No training process is currently running.", "error")
    return redirect(url_for("home"))


@app.route("/predict", methods=["POST"])
def predict():
    model_path_str = request.form.get("model_path")
    if not model_path_str: 
        return render_template("result.html", error="Please select a model.")
    if "image_file" not in request.files or request.files["image_file"].filename == "":
        return render_template("result.html", error="Please upload an image file.")

    image_file = request.files["image_file"]
    filename = secure_filename(image_file.filename)
    image_path = UPLOAD_FOLDER / filename
    image_file.save(image_path)

    try:
        model_path = MODELS_FOLDER / model_path_str
        # model_name = "effnetb4" if "effnetb4" in model_path_str else ("effnetb2" if "effnetb2" in model_path_str else "effnetb0")
        class_names = ["pizza", "steak", "sushi"]

        match = re.search(r"effnetb\d+", model_path_str)
        model_name = match.group(0) if match else "effnetb0"

        # Use the `get_pytorch_device()` function to get the correct device string
        device_str = get_pytorch_device()
        device = torch.device(device_str)

        model = model_builder.EfficientNet(model_name=model_name, num_classes=len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        img, pred_title = predictor.pred_and_plot_image(
            model=model,
            image_path=str(image_path),
            class_names=class_names,
            device=device,
            return_fig=True,
        )

        buf = io.BytesIO()
        plt.imshow(img)
        plt.title(pred_title)
        plt.axis(False)
        plt.savefig(buf, format="png")
        plt.close()
        result_image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return render_template("result.html", result_image=result_image_base64)
    except Exception as e:
        # Pass the exception to the template for debugging
        return render_template("result.html", error=f"An error occurred: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
