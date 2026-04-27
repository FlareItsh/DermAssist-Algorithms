"""
DermAssist - Inference Module
===============================
Prediction logic for single images using a trained model.

Demonstrates how to:
  1. Load a .pth checkpoint from models/production/
  2. Preprocess a PIL Image
  3. Run inference and return predicted label + confidence

Usage:
    python -m src.inference --image path/to/image.jpg
    python -m src.inference --image path/to/image.jpg --config config.yaml
"""

import os
import sys
import argparse
from typing import Tuple, Dict, Optional, Union

import yaml
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_val_transforms, load_config
from src.model import SkinLesionClassifier


# ============================================================
# Inference Engine
# ============================================================

class SkinLesionPredictor:
    """
    Loads a trained .pth model and performs inference on
    individual PIL images.
    """

    def __init__(
        self,
        model_path: str,
        config: dict,
        device: Optional[str] = None,
        architecture: str = "resnet50",
    ):
        """
        Args:
            model_path: Path to the .pth checkpoint file.
            config:     Parsed config.yaml dictionary.
            device:     "cuda", "cpu", or "auto".
        """
        # ---- Resolve device ----
        if device is None or device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ---- Load checkpoint ----
        print(f"[Inference] Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # ---- Extract class names ----
        # Try checkpoint first, fall back to config
        if "class_names" in checkpoint:
            self.class_names = checkpoint["class_names"]
        else:
            self.class_names = config["model"]["class_names"]

        num_classes = len(self.class_names)

        # ---- Build model and load weights ----
        # Priority: Checkpoint > Manual Override (architecture param) > Config
        arch = architecture
        if "architecture" in checkpoint:
            arch = checkpoint["architecture"]
        
        self.architecture = arch

        self.model = SkinLesionClassifier(
            num_classes=num_classes,
            pretrained=False,
            dropout_rate=config["model"].get("dropout_rate", 0.5),
            architecture=arch
        )
        # ---- Deep Weight Mapping (Legacy Fix) ----
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        
        # If the model has 'classifier' but checkpoint has 'backbone.fc', we bridge them
        for key, value in state_dict.items():
            if key.startswith("backbone.fc."):
                new_key = key.replace("backbone.fc.", "classifier.")
                new_state_dict[new_key] = value
            new_state_dict[key] = value
            
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        
        self.model.to(self.device)
        self.model.eval()

        # ---- Build transform pipeline ----
        self.transform = get_val_transforms(config)

        print(f"[Inference] Model loaded successfully on {self.device}")
        print(f"[Inference] Classes: {self.class_names}")

    # ---------------------------------------------------------
    # Predict a single image
    # ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, image: Union[Image.Image, str]) -> dict:
        """
        Run inference on a single PIL Image or a path to an image.

        Args:
            image: PIL Image or string path.

        Returns:
            Dictionary with prediction results.
        """
        # If a path was provided, open it
        if isinstance(image, str):
            image = Image.open(image)

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(input_batch)
        probabilities = F.softmax(logits, dim=1)

        # Extract top prediction
        confidence, class_idx = torch.max(probabilities, dim=1)
        class_idx = class_idx.item()
        confidence = confidence.item()

        # Build full probability map
        all_probs = {
            self.class_names[i]: round(probabilities[0][i].item(), 4)
            for i in range(len(self.class_names))
        }

        return {
            "label": self.class_names[class_idx],
            "confidence": round(confidence, 4),
            "class_index": class_idx,
            "all_probabilities": all_probs,
        }

    # ---------------------------------------------------------
    # Predict top-K
    # ---------------------------------------------------------
    @torch.no_grad()
    def predict_top_k(
        self, image: Image.Image, k: int = 3
    ) -> list:
        """
        Return the top-K predictions.

        Returns:
            List of dicts: [{'label': str, 'confidence': float}, ...]
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        logits = self.model(input_batch)
        probabilities = F.softmax(logits, dim=1)

        top_k = torch.topk(probabilities, k=min(k, len(self.class_names)), dim=1)
        results = []
        for i in range(top_k.values.size(1)):
            results.append({
                "label": self.class_names[top_k.indices[0][i].item()],
                "confidence": round(top_k.values[0][i].item(), 4),
            })

        return results


# ============================================================
# Factory Function
# ============================================================

def load_predictor(
    config_path: str = "config.yaml",
    model_path: Optional[str] = None,
    architecture: Optional[str] = None,
) -> SkinLesionPredictor:
    """
    Convenience function to create a SkinLesionPredictor.

    Args:
        config_path: Path to config.yaml.
        model_path:  Override model path (defaults to config value).
    """
    config = load_config(config_path)

    if model_path is None:
        model_path = config["inference"]["model_path"]

    device = config["inference"].get("device", "auto")
    arch = architecture if architecture else config["advanced"].get("active_architecture", "resnet50")

    return SkinLesionPredictor(
        model_path=model_path,
        config=config,
        device=device,
        architecture=arch
    )


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DermAssist — Run inference on a skin lesion image"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image file"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to .pth model file (overrides config)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top predictions to display"
    )
    args = parser.parse_args()

    # Load predictor
    predictor = load_predictor(
        config_path=args.config,
        model_path=args.model,
    )

    # Load image
    print(f"\n[Inference] Processing: {args.image}")
    image = Image.open(args.image)

    # Get prediction
    result = predictor.predict(image)

    # Display results
    print(f"\n{'═' * 50}")
    print(f"  PREDICTION RESULT")
    print(f"{'═' * 50}")
    print(f"  Label:       {result['label']}")
    print(f"  Confidence:  {result['confidence'] * 100:.2f}%")
    print(f"{'─' * 50}")

    # Top-K
    top_k = predictor.predict_top_k(image, k=args.top_k)
    print(f"  Top-{args.top_k} Predictions:")
    for i, pred in enumerate(top_k, 1):
        bar = "█" * int(pred["confidence"] * 30)
        print(f"    {i}. {pred['label']:25s} {pred['confidence'] * 100:6.2f}%  {bar}")
    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    main()
