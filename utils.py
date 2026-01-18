import torch
import csv

import yaml
from training_config import TrainingConfig


def write_model_checkpoint(model, epoch: int, name: str) -> None:
    torch.save(model.state_dict(), f"./{name}_checkpoint_{epoch}.pt")


def write_model_training_stats(stats: list, model_name: str) -> None:
    with open(f"model_training_stats_{model_name}.csv", "a+") as fp:
        writer = csv.writer(fp)
        writer.writerow(stats)


def load_config_file(file_name: str) -> TrainingConfig:
    with open(file_name, "r") as fp:
        conf = yaml.safe_load(fp)
    return TrainingConfig(**conf)
