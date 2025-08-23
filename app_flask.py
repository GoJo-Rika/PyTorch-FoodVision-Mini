import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Import from the new shared webapp package
from webapp import config, logic, state, utils

utils.download_sample_images()

# Loading secret key from environment variables
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.secret_key = os.getenv("SECRET_KEY")


# --- Web App Routes ---
@app.route("/", methods=["GET"])
def home():
    logic.check_training_status()
    message = request.args.get("message")
    category = request.args.get("category", "success")
    sample_images = [f.name for f in config.SAMPLES_FOLDER.glob("*.jpg")]

    return render_template(
        "index.html",
        models=logic.get_available_models(),
        training_status=state.training_log,
        device_info=logic.get_device_info(),
        sample_images=sample_images,
        message=message,
        category=category,
    )


@app.route("/status")
def status():
    just_finished, message = logic.check_training_status()
    if just_finished:
        state.training_log["message"] = message
    else:
        state.training_log.pop("message", None)  # Clean up old messages
    return jsonify(state.training_log)


@app.route("/train", methods=["POST"])
def train():
    success, message = logic.start_training_process(
        model=request.form.get("model"),
        data_name=request.form.get("data_name"),
        epochs=request.form.get("epochs"),
        lr=request.form.get("learning_rate"),
        batch_size=request.form.get("batch_size"),
    )
    flash(message, "success" if success else "error")
    return redirect(url_for("home"))


@app.route("/cancel_training", methods=["POST"])
def cancel_training():
    success, message = logic.cancel_current_training()
    flash(message, "success" if success else "error")
    return redirect(url_for("home"))

@app.route("/predict_sample", methods=["GET"])
def predict_sample():
    model_path_str = request.args.get("model_path")
    image_name = request.args.get("image_name")

    if not model_path_str or not image_name:
        return render_template("result.html", error="Model and sample image must be selected.")

    image_path = config.SAMPLES_FOLDER / image_name

    image_base64, error = logic.perform_prediction(model_path_str=model_path_str, image_path=image_path)

    return render_template("result.html", result_image=image_base64, error=error)


@app.route("/predict", methods=["POST"])
def predict():
    if "image_file" not in request.files or request.files["image_file"].filename == "":
        return render_template("result.html", error="Please upload an image file.")

    image_file = request.files["image_file"]
    filename = secure_filename(image_file.filename)

    # --- Image Format Validation ---
    if not logic.is_allowed_file(filename):
        error_msg = "Invalid file type. Please upload a PNG, JPG, JPEG, or GIF image."
        return render_template("result.html", error=error_msg)

    image_path = config.UPLOAD_FOLDER / filename
    image_file.save(image_path)

    image_base64, error = logic.perform_prediction(model_path_str=request.form.get("model_path"), image_path=image_path)

    return render_template("result.html", result_image=image_base64, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
