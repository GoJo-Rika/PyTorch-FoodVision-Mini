import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

# Import from the new shared webapp package
from webapp import config, logic, state

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
    return render_template(
        "index.html",
        models=logic.get_available_models(),
        training_status=state.training_log,
        device_info=logic.get_device_info(),
    )


@app.route("/status")
def status():
    logic.check_training_status()
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


@app.route("/predict", methods=["POST"])
def predict():
    if "image_file" not in request.files or request.files["image_file"].filename == "":
        return render_template("result.html", error="Please upload an image file.")

    image_file = request.files["image_file"]
    filename = secure_filename(image_file.filename)
    image_path = config.UPLOAD_FOLDER / filename
    image_file.save(image_path)

    image_base64, error = logic.perform_prediction(
        model_path_str=request.form.get("model_path"), image_path=image_path
    )

    return render_template("result.html", result_image=image_base64, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
