import logging
import time

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from torch.utils.data import DataLoader
from typing_extensions import Annotated
from rich.logging import RichHandler
from training_config import TrainingConfig
from utils import load_config_file, write_model_checkpoint, write_model_training_stats
from data_loader import prepare_dataset, WheatDataset

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")


def train_model(
    train_dataset: WheatDataset,
    val_dataset: WheatDataset,
    training_config: TrainingConfig,
    print_freq: int = 100,
):
    logger.info(f"Starting to train {training_config.model_name}")

    logger.info("Creating data loaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.n_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.n_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    logger.info("Done")

    logger.info(f"Creating and compiling {training_config.model_name} from timm")
    model = timm.create_model(
        training_config.model_name,
        pretrained=True,
        num_classes=training_config.n_classes,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.compile(model)
    model.to(device, memory_format=torch.channels_last)
    logger.info("Done")

    logger.info("Setting up optimizer")
    criterion = nn.CrossEntropyLoss()
    # Use the Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.optimization.learning_rate,
        weight_decay=training_config.optimization.weight_decay,
    )
    logger.info("Done")

    # Scaler is needed to scale the gradient to avoid underflow, since
    # we are training using fp16
    scaler = torch.amp.GradScaler("cuda")

    logger.info("Beginning the training loop")
    for epoch in range(training_config.num_epochs):
        # Training phase
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over the training data
        iteration = 0
        start_time = time.time()
        for inputs, labels in train_loader:
            # Move data to the specified device (GPU or CPU)

            inputs, labels = (
                inputs.to(device, memory_format=torch.channels_last, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

            # Zero the parameter gradients to prevent accumulation
            optimizer.zero_grad()
            # Forward pass
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                # Calculate the loss
                loss = criterion(outputs, labels)
            # Backward pass and optimize
            if loss is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if iteration % print_freq == 0:
                logger.info(
                    f"Epoch {epoch}: Iteration {iteration}/{len(train_loader)} - Loss: {loss.item():.4f} - Time {time.time() - start_time}"
                )
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Accumulate training loss
            running_loss += loss.item()
            iteration += 1

        # Calculate average training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Disable gradient calculation for validation
        with torch.no_grad():
            # Iterate over the validation data
            for inputs, labels in val_loader:
                # Move data to the specified device (GPU or CPU)
                inputs, labels = (
                    inputs.to(
                        device, memory_format=torch.channels_last, non_blocking=True
                    ),
                    labels.to(device, non_blocking=True),
                )
                # Forward pass
                outputs = model(inputs)
                # Calculate the loss
                loss = criterion(outputs, labels)

                # Accumulate validation loss
                val_loss += loss.item()
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                # Update total and correct predictions
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        logger.info(f"Epoch {epoch + 1}/{training_config.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        end_time = time.time()
        logger.info(f"Time for epoch = {end_time - start_time}")
        if epoch % 2 == 0:
            write_model_checkpoint(model, epoch, training_config.model_name)
        stats = [epoch, train_loss, train_acc, val_loss, val_acc, end_time - start_time]
        write_model_training_stats(stats, training_config.model_name)

    logger.info("Training completed!")


def main(
    training_list_file: Annotated[
        str, typer.Option(help="File containing list of training files")
    ],
    validation_list_file: Annotated[
        str, typer.Option(help="File containing list of validation files")
    ],
    training_config_file: Annotated[
        str, typer.Option(help="Configuration file for training")
    ],
):
    config = load_config_file(training_config_file)
    train_dataset = prepare_dataset(training_list_file, model_name=config.model_name)
    val_dataset = prepare_dataset(validation_list_file, model_name=config.model_name)

    train_model(train_dataset, val_dataset, config)


if __name__ == "__main__":
    typer.run(main)
