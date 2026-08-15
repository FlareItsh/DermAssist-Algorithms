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
from torchvision import transforms as T

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_val_transforms, load_config, HairRemoval, ColorConstancy
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
# Ensemble Predictor
# ============================================================

class EnsemblePredictor:
    """
    Loads multiple trained models and averages their predictions
    using Test-Time Augmentation (TTA) for maximum accuracy.
    """

    # TTA augmentations applied to each image at inference time.
    # Running 5 slight variations and averaging them effectively
    # gives the model more "looks" at the image, reducing noise.
    _TTA_TRANSFORMS = [
        T.Compose([]),  # Original
        T.Compose([T.RandomHorizontalFlip(p=1.0)]),  # Flipped
        T.Compose([T.RandomResizedCrop(224, scale=(0.85, 1.0))]),  # Slight zoom
        T.Compose([T.ColorJitter(brightness=0.15, contrast=0.15)]),  # Brightness shift
        T.Compose([T.RandomRotation(degrees=10)]),  # Slight rotation
    ]

    def __init__(
        self,
        model_paths_and_archs: list[Tuple[str, str]],
        config: dict,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_paths_and_archs: List of tuples (model_path, architecture).
            config: Parsed config.yaml dictionary.
            device: Target device ("cuda", "cpu", or "auto").
        """
        self.predictors = []
        for path, arch in model_paths_and_archs:
            if os.path.exists(path):
                self.predictors.append(
                    SkinLesionPredictor(
                        model_path=path,
                        config=config,
                        device=device,
                        architecture=arch,
                    )
                )
            else:
                print(f"[Ensemble] ⚠ WARNING: Model not found at {path}, skipping.")
        
        if not self.predictors:
            raise ValueError("No valid models found for the ensemble!")
            
        self.class_names = self.predictors[0].class_names
        self.architecture = f"ensemble ({', '.join([arch for _, arch in model_paths_and_archs])})"
        self.device = self.predictors[0].device
        print(f"[Ensemble] ✓ Loaded {len(self.predictors)} models with TTA ({len(self._TTA_TRANSFORMS)} augmentations).")

    def _tta_predict(self, predictor: SkinLesionPredictor, image: Image.Image) -> dict:
        """
        Run a single predictor over all TTA variants and average the probabilities.
        """
        # 1. Apply high-res preprocessing ONCE on the original image first
        preprocessed_image = image
        has_hair_removal = any(isinstance(t, HairRemoval) for t in predictor.transform.transforms)
        has_color_constancy = any(isinstance(t, ColorConstancy) for t in predictor.transform.transforms)
        
        if has_hair_removal:
            preprocessed_image = HairRemoval()(preprocessed_image)
        if has_color_constancy:
            preprocessed_image = ColorConstancy()(preprocessed_image)

        # Resize once to the standard size before TTA crops/flips
        base_image = preprocessed_image.resize((224, 224))
        
        # 2. Backup the predictor's transform and temporarily remove high-res steps
        original_transform = predictor.transform
        clean_transforms = [
            t for t in original_transform.transforms
            if not isinstance(t, (HairRemoval, ColorConstancy))
        ]
        predictor.transform = T.Compose(clean_transforms)
        
        try:
            accumulated = {cls: 0.0 for cls in self.class_names}
            for aug in self._TTA_TRANSFORMS:
                augmented = aug(base_image)
                result = predictor.predict(augmented)
                for cls in self.class_names:
                    accumulated[cls] += result["all_probabilities"][cls]

            num_augs = len(self._TTA_TRANSFORMS)
            return {cls: accumulated[cls] / num_augs for cls in self.class_names}
        finally:
            # 3. Restore the original transform
            predictor.transform = original_transform

    def predict(self, image: Union[Image.Image, str]) -> dict:
        """
        Run inference across all models and TTA variants, then average probabilities.
        """
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Accumulate TTA-averaged probabilities across all models
        avg_probs = {cls: 0.0 for cls in self.class_names}
        
        for predictor in self.predictors:
            tta_result = self._tta_predict(predictor, image)
            for cls in self.class_names:
                avg_probs[cls] += tta_result[cls]

        # Average across models
        num_models = len(self.predictors)
        for cls in self.class_names:
            avg_probs[cls] = round(avg_probs[cls] / num_models, 4)

        # Find winner
        best_class = max(avg_probs, key=avg_probs.get)
        best_confidence = avg_probs[best_class]
        class_idx = self.class_names.index(best_class)

        return {
            "label": best_class,
            "confidence": best_confidence,
            "class_index": class_idx,
            "all_probabilities": avg_probs,
        }

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
