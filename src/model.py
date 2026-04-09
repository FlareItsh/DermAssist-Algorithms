"""
DermAssist - Model Architecture
================================
ResNet50 transfer learning model for multi-class skin lesion
classification.

Architecture:
  - Pre-trained ResNet50 backbone (ImageNet weights)
  - Custom classification head with dropout regularization
  - Configurable number of output classes
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SkinLesionClassifier(nn.Module):
    """
    ResNet50-based classifier for skin lesion detection.

    The pre-trained backbone extracts rich visual features.
    A custom classification head maps those features to
    disease categories.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_classes:     Number of target classes.
            pretrained:      Whether to use ImageNet pre-trained weights.
            dropout_rate:    Dropout probability before the final layer.
            freeze_backbone: If True, freeze all backbone parameters
                             (useful for feature extraction mode).
        """
        super(SkinLesionClassifier, self).__init__()

        # ---- Backbone ----
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)

        # Number of features from the backbone's final pooling layer
        in_features = self.backbone.fc.in_features

        # ---- Classification Head ----
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes),
        )

        # Optionally freeze backbone layers
        if freeze_backbone:
            self._freeze_backbone()

        print(f"[Model] SkinLesionClassifier initialized — "
              f"{num_classes} classes, pretrained={pretrained}, "
              f"dropout={dropout_rate}")

    def _freeze_backbone(self):
        """Freeze all backbone layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("[Model] Backbone layers frozen (only FC head is trainable)")

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[Model] All backbone layers unfrozen for fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        return self.backbone(x)

    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Factory Function
# ============================================================

def build_model(config: dict, device: Optional[torch.device] = None) -> SkinLesionClassifier:
    """
    Build and return a SkinLesionClassifier from config.

    Args:
        config: Parsed config.yaml dictionary.
        device: Target device (auto-detected if None).

    Returns:
        Model moved to the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SkinLesionClassifier(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"].get("dropout_rate", 0.5),
    )

    model = model.to(device)

    total = model.get_total_params()
    trainable = model.get_trainable_params()
    print(f"[Model] Total params: {total:,} | Trainable: {trainable:,}")
    print(f"[Model] Device: {device}")

    return model


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)

    # Test forward pass with dummy input
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"[Test] Input shape:  {dummy_input.shape}")
    print(f"[Test] Output shape: {output.shape}")
    print(f"[Test] Output logits sample: {output[0].detach().cpu().numpy()}")
