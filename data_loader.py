from typing import Iterable
import pandas as pd
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import transforms

IMGNET_MEANS = [0.485, 0.456, 0.406]
IMGNET_STDS = [0.229, 0.224, 0.225]


class WheatDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_paths = []
        self.labels = []
        self.cpu_resize = T.Resize((224, 224))
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["img_path"]
        image = Image.open(img_path).convert("RGB")
        # image = F.to_tensor(image)
        # image = self.cpu_resize(image)

        label = self.df.iloc[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(mode: str, means: Iterable, stds: Iterable):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomAffine(30, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=means, std=stds),
        ]
    )
    if mode == "train":
        return train_transforms
    return val_transforms


def prepare_dataset(dataset_file: str, model_name: str) -> WheatDataset:
    if "resnet" in model_name or "regnet" in model_name or "mobile" in model_name:
        means = IMGNET_MEANS
        stds = IMGNET_STDS
    else:
        means = [0, 0, 0]
        stds = [1, 1, 1]

    if "train" in dataset_file:
        tr = get_transforms("train", means, stds)
    else:
        tr = get_transforms("val", means, stds)
    df = pd.read_json(dataset_file)
    dataset = WheatDataset(df, transform=tr)
    return dataset
