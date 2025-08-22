"""
Script to make a prediction on a single image using a trained model.

Accepts either a local file path or a URL as input.
"""
import argparse
from pathlib import Path

import requests
import torch

from food_vision import model_builder
from food_vision import predict as predictor


def main(args: argparse.Namespace) -> None:
    """
    Loads a model and makes a prediction on a specified image (from path or URL).

    Args:
        args: Command-line arguments parsed by argparse.

    """
    image_input = args.image_path
    local_image_path = ""

    # Check if the input is a URL
    if image_input.startswith("http"):
        print("Input is a URL, attempting to download...")
        try:
            # Get a filename from the URL
            image_name = Path(image_input).name
            image_dir = Path("data")
            image_dir.mkdir(parents=True, exist_ok=True)
            local_image_path = image_dir / image_name

            # Download the image
            with requests.get(image_input, stream=True) as r:
                r.raise_for_status() # Raises an exception for bad status codes
                with local_image_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded image to: {local_image_path}")

        except Exception as e:
            print(f"Error downloading image: {e}")
            return # Exit if download fails
    else:
        # It's a local file path
        local_image_path = Path(image_input)
        if not local_image_path.is_file():
            print(f"Error: File not found at '{local_image_path}'")
            return

    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hardcoded class names for this project
    class_names = ["pizza", "steak", "sushi"]

    # Create the model instance
    model = model_builder.EfficientNet(
        model_name=args.model_name,
        num_classes=len(class_names),
    )

    # Load the saved state dict
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # Make prediction and plot it
    predictor.pred_and_plot_image(
        model=model,
        image_path=str(local_image_path),
        class_names=class_names,
        device=device,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a prediction on an image from a local path or URL.")
    parser.add_argument("--image_path", type=str, required=True, help="Path or URL to the target image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint (.pth).")
    parser.add_argument("--model_name", type=str, default="effnetb0", choices=["effnetb0", "effnetb2"], help="The architecture name of the saved model.")

    args = parser.parse_args()
    main(args)
