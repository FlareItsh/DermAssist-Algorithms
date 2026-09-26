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
    Multi-architecture classifier for skin lesion detection.
    Supports ResNet50, EfficientNet-V2, and Swin Transformer.
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        architecture: str = "resnet50",
    ):
        """
        Args:
            num_classes:     Number of target classes.
            pretrained:      Whether to use ImageNet pre-trained weights.
            dropout_rate:    Dropout probability before the final layer.
            freeze_backbone: If True, freeze all backbone parameters.
            architecture:    "resnet50", "efficientnet_v2", or "swin_transformer".
        """
        super(SkinLesionClassifier, self).__init__()
        self.architecture = architecture.lower()

        # ---- Backbone Selection ----
        if self.architecture == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
        
        elif self.architecture == "efficientnet_v2":
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity() # Remove default head

        elif self.architecture == "swin_transformer":
            weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_t(weights=weights)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity() # Remove default head
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # ---- Unified Classification Head ----
        if self.architecture == "resnet50":
            # Matching the EXACT sequence of your legacy weights
            # We assign it to BOTH backbone.fc and self.classifier
            # so it can load either key name ("backbone.fc.X" or "classifier.X")
            head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout_rate * 0.5),
                nn.Linear(512, num_classes),
            )
            self.backbone.fc = head
            self.classifier = head
        else:
            # New architecture head
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(p=dropout_rate),
                nn.Linear(512, num_classes),
            )

        if freeze_backbone:
            self._freeze_backbone()

        print(f"[Model] {self.architecture.upper()} initialized — "
              f"{num_classes} classes, pretrained={pretrained}")

    def _freeze_backbone(self):
        """Freeze all backbone layers except the custom classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[Model] {self.architecture} backbone layers frozen")

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print(f"[Model] All layers unfrozen for fine-tuning")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Backbone -> Global Pooling (if needed) -> Classifier."""
        if self.architecture == "resnet50":
            # For ResNet50, the classifier is integrated into self.backbone.fc
            # so the backbone output IS the final prediction.
            return self.backbone(x)
        
        # For other architectures, we use the custom classifier head
        features = self.backbone(x)
        
        # Flatten features if they are from a CNN
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
            
        return self.classifier(features)

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
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
        architecture=config["advanced"].get("active_architecture", "resnet50"),
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
