import shutil
from typing import Annotated
from urllib.parse import quote

import matplotlib
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

matplotlib.use("Agg")

# Import from the new shared webapp package
from webapp import config, logic, state

# --- FastAPI App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Web App Routes ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request, message: str = None, category: str = "success"):
    logic.check_training_status()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": logic.get_available_models(),
            "training_status": state.training_log,
            "device_info": logic.get_device_info(),
            "message": message,
            "category": category,
        },
    )


@app.get("/status", response_class=JSONResponse)
async def status():
    just_finished, message = logic.check_training_status()
    if just_finished:
        state.training_log["message"] = message
    else:
        state.training_log.pop("message", None)
    return JSONResponse(content=state.training_log)


@app.post("/train")
async def train(
    model: Annotated[str, Form()] = ...,
    data_name: Annotated[str, Form()] = ...,
    epochs: Annotated[int, Form()] = ...,
    learning_rate: Annotated[float, Form()] = ...,
    batch_size: Annotated[int, Form()] = ...,
):
    success, message = logic.start_training_process(model, data_name, epochs, learning_rate, batch_size)
    # Redirect with a message for the user
    category = "success" if success else "error"
    return RedirectResponse(url=f"/?message={quote(message)}&category={category}", status_code=303)


@app.post("/cancel_training")
async def cancel_training():
    success, message = logic.cancel_current_training()
    category = "success" if success else "error"
    return RedirectResponse(url=f"/?message={quote(message)}&category={category}", status_code=303)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    model_path: Annotated[str, Form()] = ...,
    image_file: Annotated[UploadFile, File()] = ...,
):
    if not image_file.filename:
        return templates.TemplateResponse("result.html", {"request": request, "error": "Please upload an image file."})

    filename = image_file.filename

    # --- Image Format Validation ---
    if not logic.is_allowed_file(filename):
        error_msg = "Invalid file type. Please upload a PNG, JPG, JPEG, or GIF image."
        return templates.TemplateResponse("result.html", {"request": request, "error": error_msg})

    image_path = config.UPLOAD_FOLDER / image_file.filename
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)

    image_base64, error = logic.perform_prediction(model_path_str=model_path, image_path=image_path)

    return templates.TemplateResponse("result.html", {"request": request, "result_image": image_base64, "error": error})
