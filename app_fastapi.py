import base64
import io
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")

from typing import Annotated

import matplotlib.pyplot as plt
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

# ---- FastAPI App Initialization ----
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---- Helper Functions ----
def get_pytorch_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# This function uses the core function for display purposes
def get_device_info():
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
    just_finished = False
    message_to_flash = None
    if training_process:
        return_code = training_process.poll()
        if return_code is None:
            training_log["status"] = "Running"
        else:
            if training_log["status"] == "Running":
                just_finished = True
                model_filename = training_log.get("model_filename", "The last model")
                if return_code == 0:
                    message_to_flash = f"Training for '{model_filename}' completed successfully! It is now available."
                    training_log["details"] = f"Last run for '{model_filename}' completed successfully."
                else:
                    message_to_flash = f"Training for '{model_filename}' failed. Please check terminal."
                    training_log["details"] = f"Last run for '{model_filename}' failed."

            training_log["status"] = "Idle"
            training_process = None
    return just_finished, message_to_flash


# ---- Web App Routes ----
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str = None, category: str = "success"): # Accept message from query
    available_models = get_available_models()
    device_info = get_device_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": available_models,
            "training_status": training_log,
            "device_info": device_info,
            "message": message,
            "category": category,   # Pass message to template
        },
    )


@app.get("/status", response_class=JSONResponse)
async def status():
    just_finished, message = check_training_status()
    if just_finished:
        training_log["message"] = message
    else:
        training_log.pop("message", None)
    return JSONResponse(content=training_log)


@app.post("/train")
async def train(
    model: Annotated[str, Form()] = ...,
    data_name: Annotated[str, Form()] = ...,
    epochs: Annotated[int, Form()] = ...,
    learning_rate: Annotated[float, Form()] = ...,
    batch_size: Annotated[int, Form()] = ...,
):
    global training_process, training_log
    if training_process and training_process.poll() is None:
        message = "A training process is already running."
        return RedirectResponse(url=f"/?message={quote(message)}&category=error", status_code=303)

    model_filename = f"{model}_{data_name}_{epochs}_epochs.pth"
    training_log["model_filename"] = model_filename

    command = [
        "python", "train.py", "--model", model, "--data_name", data_name,
        "--data_url", DATASET_URLS.get(data_name), "--epochs", str(epochs),
        "--learning_rate", str(learning_rate), "--batch_size", str(batch_size),
    ]

    training_process = subprocess.Popen(command)
    training_log["status"] = "Running"
    training_log["details"] = f"Training model: {model_filename}"

    return RedirectResponse(url="/", status_code=303)


@app.post("/cancel_training")
async def cancel_training():
    global training_process, training_log
    message = "No training process is currently running."
    category = "error"
    if training_process and training_process.poll() is None:
        model_filename = training_log.get("model_filename", "the model")
        training_process.terminate()
        training_process = None
        training_log["status"] = "Idle"
        training_log["details"] = f"Training for '{model_filename}' was cancelled."
        message = "Training process has been cancelled."
        category = "success"

    return RedirectResponse(url=f"/?message={quote(message)}&category={category}", status_code=303)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    model_path: Annotated[str, Form()] = ...,
    image_file: Annotated[UploadFile, File()] = ...,
):
    if not image_file.filename:
        return templates.TemplateResponse("result.html", {"request": request, "error": "Please upload an image file."})
    image_path = UPLOAD_FOLDER / image_file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)
    try:
        full_model_path = MODELS_FOLDER / model_path
        # model_name = "effnetb4" if "effnetb4" in full_model_path.name else ("effnetb2" if "effnetb2" in full_model_path.name else "effnetb0")
        class_names = ["pizza", "steak", "sushi"]

        match = re.search(r"effnetb\d+", full_model_path.name)
        model_name = match.group(0) if match else "effnetb0"

        device = torch.device(get_pytorch_device())
        model = model_builder.EfficientNet(model_name=model_name, num_classes=len(class_names))
        model.load_state_dict(torch.load(full_model_path, map_location=device))
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

        return templates.TemplateResponse("result.html", {"request": request, "result_image": result_image_base64})
    except Exception as e:
        return templates.TemplateResponse("result.html", {"request": request, "error": f"An error occurred: {e}"})
