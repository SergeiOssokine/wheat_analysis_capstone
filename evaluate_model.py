import logging

import pandas as pd
import torch
import torch.nn as nn
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from data_loader import WheatDataset
from training_config import TrainingConfig

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")


def evaluate_model(
    model,
    dataset: WheatDataset,
    training_config: TrainingConfig,
    output_file:str|None=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Creating data loaders")
    val_loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.n_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    criterion = nn.CrossEntropyLoss()
    model.to(device, memory_format=torch.channels_last)
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    logger.info("Computing accuracy measures")
    # Disable gradient calculation for validation
    all_labels = []
    all_preds = []
    with torch.no_grad():
        # Iterate over the validation data
        for inputs, labels in val_loader:
            # Move data to the specified device (GPU or CPU)
            inputs, labels = (
                inputs.to(device, memory_format=torch.channels_last, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            # Forward pass
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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
            all_labels.extend(labels.tolist())
            all_preds.extend(predicted.tolist())
    logger.info("Done")
    # Calculate average validation loss and accuracy
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    results = pd.DataFrame({"labels":all_labels, "preds":all_preds})
    if output_file:
       
        logger.info(f"Writing labels and predictions to {output_file}")
        results.to_csv(output_file)
    return results, val_loss, val_acc
