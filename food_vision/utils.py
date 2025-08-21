"""Contains various utility functions for PyTorch model training and saving and data handling."""

import zipfile
from datetime import datetime
from pathlib import Path

import requests
import torch
from torch.utils.tensorboard import SummaryWriter


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="foodvision_mini_efficient_b0_model.pth")

    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith((".pth", ".pt")), ("model_name should end with '.pt' or '.pth'")
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def create_writer(experiment_name: str, model_name: str, extra: str = None) -> SummaryWriter:
    """Creates a TensorBoard SummaryWriter with a structured log directory."""
    # Get timestamp of current date in reverse order
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Create log directory path
    log_dir = Path("runs") / timestamp / experiment_name / model_name

    if extra:
        log_dir /= extra
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")
    return SummaryWriter(log_dir=str(log_dir))


def download_and_unzip_data(source_url: str, destination_name: str) -> Path:
    """
    Downloads and unzips data from a given URL if not already present locally.

    Args:
        source_url (str): URL to download the zip file from.
        destination_name (str): Name of the directory to extract the data to.

    Returns:
        Path: Path to the extracted data directory.

    """
    # Define the base data directory and the target extraction directory
    data_path = Path("data/")
    image_path = data_path / destination_name

    # Check if the data has already been downloaded and extracted
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory already exists, skipping download.")
    else:
        # Create the extraction directory
        print(f"[INFO] Creating {image_path} directory.")
        image_path.mkdir(parents=True, exist_ok=True)

        # Get the filename from the source URL
        target_file = Path(source_url).name

        # Download the file from the source URL
        with (data_path / target_file).open("wb") as f:
            request = requests.get(source_url, timeout=30)
            print(f"[INFO] Downloading {target_file}...")
            f.write(request.content)

        # Unzip the downloaded file into the extraction directory
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file}...")
            zip_ref.extractall(image_path)

        # Remove the zip file after extraction to save space
        (data_path / target_file).unlink()

    # Return the path to the extracted data directory
    return image_path
