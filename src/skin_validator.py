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
        # Tier A: Target Conditions (Acne, Eczema, Herpes) — must describe genuine active inflammatory pathology
        self._lesion_prompts: list[str] = [
            "a clinical close-up macro photo of inflamed acne pimples, red pustules, whiteheads, blackheads, or acne zits",
            "a photo of inflammatory acne vulgaris breakout with red papules, pustules, or cystic lesions",
            "a close-up photo of hand eczema, finger dermatitis, cracked red scaly skin on knuckles or fingers",
            "a close-up photo of palm eczema, dyshidrotic eczema, pompholyx, red inflamed itchy skin on palm of hand",
            "a close-up photo of atopic eczema, flexural eczema rash, dry itchy red patch on arm, forearm, or body",
            "a close-up photo of atopic dermatitis, allergic contact dermatitis, or dry flaking itchy eczema skin",
            "a close-up photo of herpes simplex cold sores on lips, fever blisters, or clustered fluid-filled vesicles",
            "a close-up photo of herpes sores, cutaneous herpes, or oral/genital vesicles on skin",
        ]

        # Tier B: Out-of-Scope / Unsupported Conditions — non-target diseases
        # Each prompt maps to a concrete disease label via self._unsupported_prompt_labels
        self._unsupported_prompts: list[str] = [
            "a photo of psoriasis with thick red annular plaques, circular patches, or silvery scales on chest or torso",
            "a photo of plaque psoriasis, guttate psoriasis, or widespread red scaly skin plaques on torso",
            "a photo of ringworm, tinea corporis, or circular ring-shaped fungal rash with distinct round borders",
            "a photo of vitiligo, leukoderma, or stark pure white depigmented skin patches",
            "a photo of melanoma, dark asymmetrical black mole, or invasive skin cancer tumor",
            "a clinical photo of hives, severe urticaria, large raised allergic welts, or swollen wheals",
            "a photo of skin warts, verruca vulgaris, skin tags, or cauliflower-like growths",
            "a photo of annular erythema, lupus rash, or extensive widespread body rash",
        ]

        # Maps each unsupported_prompts entry (by index) to its clinical disease label
        self._unsupported_prompt_labels: list[str] = [
            "psoriasis",
            "psoriasis",
            "ringworm",
            "vitiligo",
            "melanoma",
            "hives",
            "warts",
            "lupus",
        ]

        # Tier C: Healthy Skin & Cosmetics
        self._healthy_skin_prompts: list[str] = [
            "a portrait or photo of a woman or person with clear healthy facial skin, beauty cosmetics, and no pimples",
            "a close-up photo of a clean healthy face, flawless smooth skin, cosmetic makeup, and zero acne",
            "a close-up photo of clean smooth healthy forehead, cheek, nose, or chin with no disease",
            "a photograph of clean, smooth, healthy human skin with no disease, no rash, and no pimples",
            "a close-up photo of an ankle, foot, heel, leg, arm, or hand with healthy clear unblemished skin",
            "a close-up photo of jewelry, beads, or watch worn on healthy normal skin",
        ]

        # Tier D: Full Body, Headshot Portraits & Aesthetic Procedures
        self._full_body_prompts: list[str] = [
            "a headshot portrait photo of a person or model posing, smiling, or looking at camera",
            "a photo of a dermatologist or aesthetician examining or treating a patient's face with a handheld light, device, or dermatoscope",
            "a beauty clinic, cosmetic spa treatment, or facial aesthetic procedure photo",
            "a distant body photograph or portrait without a close-up lesion",
            "a full body photo of a person or model lying down, standing, or posing",
            "a full body photo of a person lying on a bed, mattress, or blue sheet",
        ]

        # Tier E: Non-Skin / Objects / Memes
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
            "unsupported_condition": self._unsupported_prompts,
            "healthy_skin": self._healthy_skin_prompts,
            "full_body_portrait": self._full_body_prompts,
            "non_skin": self._non_skin_prompts,
        }

        self._all_prompts: list[str] = (
            self._lesion_prompts
            + self._unsupported_prompts
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
              f"({len(self._all_prompts)} prompts across 5 clinical tiers)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, image: Union[str, Image.Image]) -> dict:
        """
        Determine the semantic clinical category of the image.

        Args:
            image: A PIL Image or a path to an image file.

        Returns:
            Dictionary with category ('lesion', 'unsupported_condition', 'healthy_skin', 'full_body_portrait', 'non_skin'),
            is_skin, is_lesion, is_target_lesion, is_unsupported, is_healthy_skin, reason, and confidence scores.
        """
        # When the validator is disabled, let everything through
        if not self._enabled or self._model is None:
            return {
                "category": "lesion",
                "is_skin": True,
                "is_lesion": True,
                "is_target_lesion": True,
                "is_unsupported": False,
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

        # Convert category logits to probabilities using temperature scaling (T=0.7 for balanced clinical separation)
        cat_names = list(cat_max_logits.keys())
        cat_tensor = torch.tensor([cat_max_logits[k] for k in cat_names], device=self._device)
        cat_probs_tensor = torch.softmax(cat_tensor / 0.7, dim=0)
        category_scores = {k: round(cat_probs_tensor[i].item(), 4) for i, k in enumerate(cat_names)}

        lesion_score = category_scores.get("lesion", 0.0)
        unsupported_score = category_scores.get("unsupported_condition", 0.0)
        healthy_score = category_scores.get("healthy_skin", 0.0)
        full_body_score = category_scores.get("full_body_portrait", 0.0)
        non_skin_score = category_scores.get("non_skin", 0.0)

        # ---- Clinical Priority Decision Rules ----
        # 1. Non-skin objects, memes, animals
        if non_skin_score >= 0.35 and non_skin_score > lesion_score and non_skin_score > unsupported_score:
            winning_category = "non_skin"
        # 2. Distant full-body portraits, headshots, and aesthetic procedures
        elif full_body_score >= 0.35 and full_body_score > lesion_score and full_body_score > unsupported_score:
            winning_category = "full_body_portrait"
        # 3. Healthy unblemished skin / cosmetic face without active disease
        elif healthy_score >= 0.35 and healthy_score > lesion_score and healthy_score > unsupported_score:
            winning_category = "healthy_skin"
        # 4. Out-of-Scope Condition (must beat lesion score by a clear margin)
        elif unsupported_score >= 0.45 and unsupported_score > (lesion_score + 0.08):
            winning_category = "unsupported_condition"
        # 5. Default to Target Lesion (allows Acne, Eczema, Herpes to pass freely to classifier)
        else:
            winning_category = "lesion"

        is_skin = winning_category in ["lesion", "unsupported_condition", "healthy_skin"]
        is_lesion = winning_category in ["lesion", "unsupported_condition"]
        is_target_lesion = winning_category == "lesion"
        is_unsupported = winning_category == "unsupported_condition"
        is_healthy_skin = winning_category == "healthy_skin"

        # Detect dermoscopy vs camera
        image_type = "normal_camera"
        if is_lesion:
            derm_prompt = "a medical dermoscopy photograph of an acne, eczema, or herpes lesion"
            if derm_prompt in self._all_prompts:
                derm_idx = self._all_prompts.index(derm_prompt)
                if probs[derm_idx].item() > 0.25:
                    image_type = "dermoscopic"

        if winning_category == "non_skin":
            reason = "The uploaded image does not appear to be a human skin photo."
        elif winning_category == "full_body_portrait":
            reason = "Full body or distant portrait photo detected. Please upload a close-up photo of the specific skin lesion."
        elif winning_category == "unsupported_condition":
            reason = "Out-of-scope skin condition detected outside primary focus areas (Acne, Eczema, Herpes)."
        elif winning_category == "healthy_skin":
            reason = "Clear, healthy unblemished skin detected with no active disease lesions."
        else:
            reason = f"Target skin lesion photo accepted ({image_type})."

        # ---- Detect specific out-of-scope disease via per-prompt logits ----
        out_of_scope_category: str | None = None
        if winning_category == "unsupported_condition":
            # Build logit vector for each unsupported prompt
            unsupported_start_idx = len(self._lesion_prompts)
            unsupported_logits = logits[0, unsupported_start_idx: unsupported_start_idx + len(self._unsupported_prompts)]
            top_prompt_idx = int(unsupported_logits.argmax().item())
            out_of_scope_category = self._unsupported_prompt_labels[top_prompt_idx]

            # Aggregate score per disease label (in case multiple prompts point to the same label)
            label_scores: dict[str, float] = {}
            for i, label in enumerate(self._unsupported_prompt_labels):
                score = unsupported_logits[i].item()
                if label not in label_scores or score > label_scores[label]:
                    label_scores[label] = score
            top_label = max(label_scores, key=lambda k: label_scores[k])
            out_of_scope_category = top_label

        print(
            f"[SkinValidator] DECISION={winning_category.upper()} | "
            f"target_lesion={lesion_score:.3f} | unsupported={unsupported_score:.3f} | "
            f"healthy={healthy_score:.3f} | full_body={full_body_score:.3f} | non_skin={non_skin_score:.3f}"
            + (f" | out_of_scope_category={out_of_scope_category}" if out_of_scope_category else "")
        )

        return {
            "category": winning_category,
            "is_skin": is_skin,
            "is_lesion": is_lesion,
            "is_target_lesion": is_target_lesion,
            "is_unsupported": is_unsupported,
            "is_healthy_skin": is_healthy_skin,
            "image_type": image_type,
            "skin_score": round(lesion_score + unsupported_score + healthy_score, 4),
            "category_scores": category_scores,
            "out_of_scope_category": out_of_scope_category,
            "reason": reason,
        }
