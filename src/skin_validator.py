"""
DermAssist - Skin Image Validator
===================================
Uses OpenAI CLIP (via HuggingFace Transformers) to perform zero-shot
classification before the main disease classifier runs.

This gate ensures that only genuine skin/lesion photos are processed,
and rejects unrelated images (animals, objects, landscapes, etc.)
regardless of skin tone — CLIP understands semantic content, not
just color ranges.

Usage:
    from src.skin_validator import SkinValidator

    validator = SkinValidator(config)
    result = validator.validate(image_path)
    if not result["is_skin"]:
        raise HTTPException(422, result["reason"])
"""

from __future__ import annotations

import os
from typing import Union

from PIL import Image


# ============================================================
# Lazy imports — only load transformers when the validator is
# actually instantiated, so the API starts fast even when the
# CLIP model has not been downloaded yet.
# ============================================================

_CLIP_AVAILABLE = True
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
except ImportError:
    _CLIP_AVAILABLE = False


class SkinValidator:
    """
    Zero-shot skin image gate using CLIP.

    Compares the uploaded image against a set of skin-related prompts
    and non-skin prompts. If the image scores higher on the non-skin
    side (or fails to meet the skin confidence threshold), it is
    rejected before ever reaching the disease classifier.

    Works across all human skin tones because CLIP reasons about
    semantic content, not pixel color distributions.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Parsed config.yaml dictionary. The ``validation``
                    key drives all behaviour.
        """
        self._config = config.get("validation", {})
        self._enabled = self._config.get("skin_validator_enabled", True)

        if not self._enabled:
            print("[SkinValidator] ⚠ Validator is DISABLED in config.")
            self._model = None
            self._processor = None
            return

        if not _CLIP_AVAILABLE:
            print(
                "[SkinValidator] ❌ `transformers` package not found. "
                "Run: pip install transformers\n"
                "[SkinValidator] ⚠ Validator will be DISABLED."
            )
            self._enabled = False
            self._model = None
            self._processor = None
            return

        model_name: str = self._config.get("clip_model", "openai/clip-vit-base-patch32")
        self._threshold: float = self._config.get("skin_confidence_threshold", 0.60)

        self._skin_prompts: list[str] = self._config.get("skin_prompts", [
            # Normal camera
            "a close-up photo of human skin with a rash or lesion",
            "a skin disease on a human body",
            "a photo of skin inflammation, wound, or discoloration on a person",
            "a macro photograph of a skin condition taken with a smartphone",
            # Dermoscopic
            "a dermoscopy image of a skin lesion with a circular dark border",
            "a dermatoscope image showing skin pigmentation and lesion structure",
            "a medical dermoscopy photograph of a mole or skin growth",
            "a polarized light dermoscopy image of a skin condition",
        ])

        self._non_skin_prompts: list[str] = self._config.get("non_skin_prompts", [
            "a photo of an object, animal, food, or outdoor scene",
            "a photo of a document, screen, or text",
            "a photo of a vehicle, building, or landscape",
            "a selfie or portrait photo of a person's face",
            "an x-ray, MRI, or internal medical scan",
        ])

        # Track which skin prompts are dermoscopy-specific for image-type detection
        self._dermoscopy_prompt_indices: list[int] = [
            i for i, p in enumerate(self._skin_prompts)
            if "dermoscop" in p.lower() or "dermatoscop" in p.lower()
        ]

        self._all_prompts: list[str] = self._skin_prompts + self._non_skin_prompts
        self._num_skin_prompts: int = len(self._skin_prompts)

        print(f"[SkinValidator] ⏳ Loading CLIP model: {model_name}")

        # Determine device — prefer GPU, fall back to CPU
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        print(f"[SkinValidator] ✅ CLIP loaded on {self._device} — "
              f"{self._num_skin_prompts} skin prompts / "
              f"{len(self._non_skin_prompts)} non-skin prompts")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, image: Union[str, Image.Image]) -> dict:
        """
        Determine whether ``image`` is a genuine skin / lesion photo.

        Args:
            image: A PIL Image or a path to an image file.

        Returns:
            A dictionary with the following keys:

            * ``is_skin`` (bool): True if the image passes the gate.
            * ``skin_score`` (float): Aggregated probability that the
              image matches the skin prompts (0.0 – 1.0).
            * ``reason`` (str): Human-readable explanation (useful for
              surfacing error messages to the user).
            * ``prompt_scores`` (dict): Per-prompt probability scores.
        """
        # When the validator is disabled, let everything through
        if not self._enabled or self._model is None:
            return {
                "is_skin": True,
                "skin_score": 1.0,
                "reason": "Skin validator is disabled.",
                "prompt_scores": {},
            }

        # ---- Load image ----
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # ---- Run CLIP ----
        inputs = self._processor(
            text=self._all_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Shape: (1, num_prompts)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0]

        # ---- Aggregate skin score ----
        # Sum probability across all skin-related prompts
        skin_probs = probs[: self._num_skin_prompts]
        skin_score: float = skin_probs.sum().item()

        # ---- Detect image type (dermoscopic vs normal camera) ----
        # Compare average score of dermoscopy prompts vs normal-camera prompts
        derm_indices = self._dermoscopy_prompt_indices
        normal_indices = [
            i for i in range(self._num_skin_prompts)
            if i not in derm_indices
        ]

        image_type = "unknown"
        if derm_indices and normal_indices:
            derm_score = probs[derm_indices].mean().item()
            normal_score = probs[normal_indices].mean().item()
            image_type = "dermoscopic" if derm_score > normal_score else "normal_camera"
        elif derm_indices:
            image_type = "dermoscopic"
        elif normal_indices:
            image_type = "normal_camera"

        # Per-prompt breakdown for debugging / logging
        prompt_scores: dict[str, float] = {
            prompt: round(probs[i].item(), 4)
            for i, prompt in enumerate(self._all_prompts)
        }

        # ---- Decision ----
        is_skin: bool = skin_score >= self._threshold

        reason: str
        if is_skin:
            reason = (
                f"Image accepted as a {image_type.replace('_', ' ')} skin photo "
                f"(skin score: {skin_score:.2%})."
            )
        else:
            reason = (
                "The uploaded image does not appear to be a skin or lesion photo. "
                "Please upload a clear, close-up photo of the affected skin area "
                "(normal camera or dermoscope both accepted)."
            )

        print(
            f"[SkinValidator] type={image_type} | "
            f"skin_score={skin_score:.4f} | "
            f"accepted={is_skin}"
        )

        return {
            "is_skin": is_skin,
            "skin_score": round(skin_score, 4),
            "image_type": image_type,
            "reason": reason,
            "prompt_scores": prompt_scores,
        }
