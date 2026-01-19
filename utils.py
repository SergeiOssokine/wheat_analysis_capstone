import torch
import csv
import timm
import yaml
from training_config import TrainingConfig


def write_model_checkpoint(
    model, optimizer, stats, training_config: TrainingConfig
) -> None:
    epoch, train_loss, _, val_loss, _, _ = stats
    checkpoint = {
        "training_config": training_config.model_dump(),
        "model_state_dict": model.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"./{training_config.model_name}_checkpoint_{epoch}.pt")


def reload_model_for_inference(checkpoint_name: str):
    checkpoint = torch.load(checkpoint_name, weights_only=True)
    tr_config = TrainingConfig(**checkpoint["training_config"])
    model_trained = timm.create_model(
        tr_config.model_name, pretrained=True, num_classes=8
    )
    model_trained = torch.compile(model_trained)
    model_trained.load_state_dict(checkpoint["model_state_dict"])
    model_trained.eval()
    return model_trained, tr_config


def save_model_to_onnx(model, model_name: str, device='cuda'):
    onnx_path = f"{model_name}.onnx"
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
    )


def write_model_training_stats(stats: list, model_name: str) -> None:
    with open(f"model_training_stats_{model_name}.csv", "a+") as fp:
        writer = csv.writer(fp)
        writer.writerow(stats)


def load_config_file(file_name: str) -> TrainingConfig:
    with open(file_name, "r") as fp:
        conf = yaml.safe_load(fp)
    return TrainingConfig(**conf)
