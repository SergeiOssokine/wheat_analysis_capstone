import logging

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from data_loader import (
    WheatDataset,
)
from training_config import TrainingConfig

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")


def evaluate_model(
    model,
    dataset: WheatDataset,
    training_config: TrainingConfig,
    output_file: str | None = None,
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
    results = pd.DataFrame({"labels": all_labels, "preds": all_preds})
    if output_file:
        logger.info(f"Writing labels and predictions to {output_file}")
        results.to_csv(output_file)
    return results, val_loss, val_acc


def get_grad_cam(model, test_image_file, img_transforms, target_layers, device="cuda"):
    image = Image.open(test_image_file)

    tmp = img_transforms(image)
    input_tensor = tmp.unsqueeze(0).to(device, memory_format=torch.channels_last)
    with torch.no_grad():
        predicted_class = model(input_tensor).argmax()

    logger.info(f"The model predicted class {predicted_class}")
    cam = GradCAM(model=model, target_layers=target_layers)
    original_image = cv2.imread(test_image_file)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Resize to match model input size
    rgb_img = cv2.resize(original_image, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    # Generate CAM
    targets = [ClassifierOutputTarget(predicted_class)]  # or None for top prediction
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        rgb_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_MAGMA
    )
    return visualization


def compute_accuracy_measures(df_trained, average="micro"):
    precision = precision_score(
        df_trained["labels"], df_trained["preds"], average=average
    )
    recall = recall_score(df_trained["labels"], df_trained["preds"], average=average)
    f1 = f1_score(df_trained["labels"], df_trained["preds"], average=average)
    print(f"precision: {precision} - recall: {recall} - f1_score: {f1}")

    return [precision, recall, f1]
