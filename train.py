"""
Main script to train a PyTorch model for image classification.

This script allows for running experiments with different models, datasets,
and hyperparameters via command-line arguments. It leverages the modular
components from the `foodvision` package.
"""
import argparse
import os

import torch
import torchvision

from food_vision import data_setup, engine, model_builder, utils


def main(args: argparse.Namespace) -> None:
    """
    Sets up and runs a single training experiment.

    Args:
        args: Command-line arguments parsed by argparse.

    """
    # ---- Setup ----
    # Set up device-agnostic code, including Apple Silicon (MPS)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")

    # ---- Download Data ----
    data_path = utils.download_and_unzip_data(
        source_url=args.data_url,
        destination_name=args.data_name,
    )
    train_dir = data_path / "train"
    # The test set is consistent across experiments (10% version)
    test_dir = data_path / "test"

    # ---- Create DataLoaders ----
    # Get the appropriate transforms for the selected model
    weights = getattr(torchvision.models, f"EfficientNet_{args.model[6:].upper()}_Weights").DEFAULT
    auto_transforms = weights.transforms()

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=auto_transforms,
        batch_size=args.batch_size,
    )

    # ---- Create Model ----
    model = model_builder.EfficientNet(
        model_name=args.model,
        num_classes=len(class_names),
    ).to(device)

    # ---- Setup Loss, Optimizer, and TensorBoard Writer ----
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = utils.create_writer(
        experiment_name=args.data_name,
        model_name=args.model,
        extra=f"{args.epochs}_epochs",
    )

    # ---- Start Training ----
    print(f"[INFO] Starting training for {args.model} on {args.data_name} for {args.epochs} epochs.")
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        writer=writer,
    )

    # ---- Save the Trained Model ----
    save_filepath = f"{args.model}_{args.data_name}_{args.epochs}_epochs.pth"
    utils.save_model(model=model, target_dir="models", model_name=save_filepath)
    print("-" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EfficientNet model on custom data.")

    parser.add_argument("--model", type=str, default="effnetb0", choices=["effnetb0", "effnetb2", "effnetb4"], help="Model architecture to use.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--data_name", type=str, default="pizza_steak_sushi_10_percent", help="Name of the data directory.")
    parser.add_argument("--data_url", type=str, default="https://github.com/GoJo-Rika/PyTorch-FoodVision-Mini/raw/main/data/pizza_steak_sushi.zip", help="URL to download the data from.")

    args = parser.parse_args()
    main(args)
