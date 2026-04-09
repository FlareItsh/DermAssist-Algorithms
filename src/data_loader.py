"""
DermAssist - Data Loader Module
================================
Handles image preprocessing, augmentation, and dataset creation
for skin lesion classification.

Supports:
  - Resizing to 224x224 (ResNet50 input)
  - Normalization using ImageNet statistics
  - Data augmentation (horizontal flips, rotation, color jitter)
  - Train/validation splitting
"""

import os
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ============================================================
# Configuration Loader
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Image Transforms
# ============================================================

def get_train_transforms(config: dict) -> transforms.Compose:
    """
    Build the training transform pipeline.

    Includes data augmentation (random flips, rotation, color jitter)
    followed by normalization with ImageNet statistics.
    """
    img_size = config["data"]["image_size"]
    mean = config["data"]["mean"]
    std = config["data"]["std"]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(config: dict) -> transforms.Compose:
    """
    Build the validation/inference transform pipeline.

    No augmentation — only resize and normalize.
    """
    img_size = config["data"]["image_size"]
    mean = config["data"]["mean"]
    std = config["data"]["std"]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ============================================================
# Dataset
# ============================================================

class SkinLesionDataset(Dataset):
    """
    Custom PyTorch Dataset for skin lesion images.

    Expected directory layout:
        data_dir/
        ├── class_0/
        │   ├── img001.jpg
        │   └── ...
        ├── class_1/
        │   ├── img001.jpg
        │   └── ...
        └── ...

    Each subdirectory name is treated as a class label.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[list] = None,
    ):
        """
        Args:
            data_dir:    Root directory containing class subdirectories.
            transform:   Torchvision transform pipeline.
            class_names: Optional ordered list of class names. If None,
                         classes are inferred from subdirectory names.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Discover classes from folder names
        if class_names is not None:
            self.classes = class_names
        else:
            self.classes = sorted([
                d.name for d in self.data_dir.iterdir() if d.is_dir()
            ])

        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # Collect all image paths and their labels
        self.samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

        print(f"[DataLoader] Loaded {len(self.samples)} images "
              f"across {len(self.classes)} classes from '{data_dir}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# DataLoader Factory
# ============================================================

def create_dataloaders(
    config: dict,
    data_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, list]:
    """
    Create training and validation DataLoaders.

    Args:
        config:   Parsed config.yaml dictionary.
        data_dir: Override for the data directory path.

    Returns:
        (train_loader, val_loader, class_names)
    """
    if data_dir is None:
        data_dir = config["data"]["raw_dir"]

    class_names = config["model"].get("class_names", None)
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    train_split = config["data"]["train_split"]

    # Build transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    # Build full dataset (using validation transforms initially for splitting)
    full_dataset = SkinLesionDataset(
        data_dir=data_dir,
        transform=None,  # We'll apply transforms after splitting
        class_names=class_names,
    )
    discovered_classes = full_dataset.classes

    # Split into train / validation
    total = len(full_dataset)
    train_size = int(total * train_split)
    val_size = total - train_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Wrap subsets with appropriate transforms
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Train: {train_size} samples | Val: {val_size} samples")
    return train_loader, val_loader, discovered_classes


class TransformSubset(Dataset):
    """
    Wraps a torch Subset to apply a specific transform pipeline.

    This allows train and validation splits to use different augmentations.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    print("Train Transforms:", get_train_transforms(cfg))
    print("Val Transforms:  ", get_val_transforms(cfg))
    print(f"Config loaded — expecting {cfg['model']['num_classes']} classes")
