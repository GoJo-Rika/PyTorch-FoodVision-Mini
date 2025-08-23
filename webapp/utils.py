from pathlib import Path

import requests
from tqdm import tqdm

from webapp import config


def download_sample_images():
    """Downloads sample images to the static folder if they don't already exist."""
    config.SAMPLES_FOLDER.mkdir(exist_ok=True)

    for name, url in config.SAMPLE_IMAGE_URLS.items():
        image_path = config.SAMPLES_FOLDER / name
        if not image_path.is_file():
            try:
                print(f"Downloading sample image: {name}...")
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with image_path.open("wb") as f:
                        for chunk in tqdm(
                            r.iter_content(chunk_size=8192), desc=f"Downloading {name}"):
                            f.write(chunk)
                print(f"Successfully downloaded {name}.")
            except Exception as e:
                print(f"Error downloading {name}: {e}")
