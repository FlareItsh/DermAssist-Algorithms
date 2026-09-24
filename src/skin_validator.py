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
        self._threshold: float = self._config.get("skin_confidence_threshold", 0.80)

        # Category-specific prompt suites
        self._lesion_prompts: list[str] = [
            "a macro medical close-up photo of acne pimples, pustules, whiteheads, or blackheads on skin",
            "a clinical close-up photo of red inflamed eczema dermatitis, scaly rash, flaking skin, or sores",
            "a clinical close-up photo of herpes blisters, cold sores, or fluid-filled vesicles on skin",
            "a clinical close-up photograph of an active diseased skin lesion, rash, or ulcer",
            "a medical dermoscopy photograph of a diseased skin lesion or abnormal mole",
        ]

        self._healthy_skin_prompts: list[str] = [
            "a close-up photo of an ankle, foot, heel, leg, arm, or hand with healthy clear skin",
            "a close-up photo of an anklet bracelet, beads, jewelry, or watch worn on healthy unblemished skin",
            "a photograph of clean, smooth, healthy human skin with no disease, no rash, and no pimples",
            "a photo of normal clear human skin with zero redness and zero lesions",
        ]

        self._full_body_prompts: list[str] = [
            "a distant body photograph or portrait without a close-up lesion",
            "a full body photo of a person or model lying down, standing, or posing",
            "a full body photo of a person lying on a bed, mattress, or blue sheet",
            "a nude woman or torso lying down horizontally on a mat or bed",
            "a portrait photograph, swimsuit photo, or distant body picture",
        ]

        self._non_skin_prompts: list[str] = [
            "a photo of an everyday object, household item, cloth, furniture, gadget, or footwear",
            "a photo of an animal, dog, cat, pet, or animal fur",
            "a photo of food, plant, flower, fruit, or vegetable",
            "a screenshot, digital document, meme, wallpaper, icon, or text",
            "a drawing, illustration, cartoon, clipart, anime, or digital graphic",
            "a photo of a vehicle, building, road, room interior, or landscape",
            "an abstract texture, pattern, wood, metal, or fabric",
        ]

        self._prompt_categories = {
            "lesion": self._lesion_prompts,
            "healthy_skin": self._healthy_skin_prompts,
            "full_body_portrait": self._full_body_prompts,
            "non_skin": self._non_skin_prompts,
        }

        self._all_prompts: list[str] = (
            self._lesion_prompts
            + self._healthy_skin_prompts
            + self._full_body_prompts
            + self._non_skin_prompts
        )

        print(f"[SkinValidator] ⏳ Loading CLIP model: {model_name}")

        # Determine device — prefer GPU, fall back to CPU
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        print(f"[SkinValidator] ✅ Multi-Category Semantic Gate loaded on {self._device} "
              f"({len(self._all_prompts)} prompts across 4 clinical tiers)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, image: Union[str, Image.Image]) -> dict:
        """
        Determine the semantic clinical category of the image.

        Args:
            image: A PIL Image or a path to an image file.

        Returns:
            Dictionary with category ('lesion', 'healthy_skin', 'full_body_portrait', 'non_skin'),
            is_skin, is_lesion, is_healthy_skin, reason, and confidence scores.
        """
        # When the validator is disabled, let everything through
        if not self._enabled or self._model is None:
            return {
                "category": "lesion",
                "is_skin": True,
                "is_lesion": True,
                "is_healthy_skin": False,
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
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0]

        # ---- Category Logit Pooling & Temperature Softmax ----
        idx = 0
        cat_max_logits: dict[str, float] = {}
        for cat_name, prompts in self._prompt_categories.items():
            count = len(prompts)
            # Take the maximum matching logit for each category
            cat_slice = logits[0, idx : idx + count]
            cat_max_logits[cat_name] = cat_slice.max().item()
            idx += count

        # Convert category logits to probabilities using temperature scaling (T=0.6 for ultra sharp separation)
        cat_names = list(cat_max_logits.keys())
        cat_tensor = torch.tensor([cat_max_logits[k] for k in cat_names], device=self._device)
        cat_probs_tensor = torch.softmax(cat_tensor / 0.6, dim=0)
        category_scores = {k: round(cat_probs_tensor[i].item(), 4) for i, k in enumerate(cat_names)}

        lesion_score = category_scores.get("lesion", 0.0)
        healthy_score = category_scores.get("healthy_skin", 0.0)
        full_body_score = category_scores.get("full_body_portrait", 0.0)
        non_skin_score = category_scores.get("non_skin", 0.0)

        # ---- Strict Clinical Priority Decision Rules ----
        if non_skin_score >= 0.30:
            winning_category = "non_skin"
        elif full_body_score >= 0.35:
            winning_category = "full_body_portrait"
        elif healthy_score >= 0.35 and lesion_score < 0.60:
            winning_category = "healthy_skin"
        elif lesion_score >= 0.60 and lesion_score > healthy_score and lesion_score > full_body_score and lesion_score > non_skin_score:
            winning_category = "lesion"
        else:
            winning_category = max(category_scores, key=category_scores.get)

        is_skin = winning_category in ["lesion", "healthy_skin"]
        is_lesion = winning_category == "lesion"
        is_healthy_skin = winning_category == "healthy_skin"

        # Detect dermoscopy vs camera
        image_type = "normal_camera"
        if is_lesion:
            derm_prompt = "a medical dermoscopy photograph of a skin lesion, mole, or skin growth"
            if derm_prompt in self._all_prompts:
                derm_idx = self._all_prompts.index(derm_prompt)
                if probs[derm_idx].item() > 0.25:
                    image_type = "dermoscopic"

        if winning_category == "non_skin":
            reason = "The uploaded image does not appear to be a human skin photo."
        elif winning_category == "full_body_portrait":
            reason = "Full body or distant portrait photo detected. Please upload a close-up photo of the specific skin lesion."
        elif winning_category == "healthy_skin":
            reason = "Clear, healthy unblemished skin detected with no active disease lesions."
        else:
            reason = f"Active skin lesion photo accepted ({image_type})."

        print(
            f"[SkinValidator] DECISION={winning_category.upper()} | "
            f"lesion={lesion_score:.3f} | healthy={healthy_score:.3f} | "
            f"full_body={full_body_score:.3f} | non_skin={non_skin_score:.3f}"
        )

        return {
            "category": winning_category,
            "is_skin": is_skin,
            "is_lesion": is_lesion,
            "is_healthy_skin": is_healthy_skin,
            "image_type": image_type,
            "skin_score": round(lesion_score + healthy_score, 4),
            "category_scores": category_scores,
            "reason": reason,
        }
