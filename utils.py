import torch
import csv

import yaml
from training_config import TrainingConfig


def write_model_checkpoint(model, optimizer,stats, name: str) -> None:
    epoch,train_loss,_,val_loss,_,_ = stats
    checkpoint = {
        'model_state_dict':model.state_dict(),
        'train_loss':train_loss,
        'val_loss':val_loss,
        'epoch':epoch,
        'optimizer_state_dict':optimizer.state_dict()
    }
    torch.save(checkpoint, f"./{name}_checkpoint_{epoch}.pt")


def write_model_training_stats(stats: list, model_name: str) -> None:
    with open(f"model_training_stats_{model_name}.csv", "a+") as fp:
        writer = csv.writer(fp)
        writer.writerow(stats)


def load_config_file(file_name: str) -> TrainingConfig:
    with open(file_name, "r") as fp:
        conf = yaml.safe_load(fp)
    return TrainingConfig(**conf)
