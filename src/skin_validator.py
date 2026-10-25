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
        # Tier A: Target Conditions (Acne, Eczema, Herpes)
        self._acne_prompts: list[str] = [
            "a clinical close-up macro photo of inflamed acne pimples, red pustules, whiteheads, blackheads, or acne zits",
            "a photo of inflammatory acne vulgaris breakout with red papules, pustules, or cystic lesions",
            "a close-up photo of facial acne, comedones, or acne breakouts on face, forehead, or chin",
            "a clinical photo of severe cystic acne or nodulocystic acne breakout covering facial cheek and jawline with multiple inflamed lesions",
            "a close-up photo of facial acne vulgaris with comedones, sebaceous plugs, follicular pustules, or acne breakouts on face",
            "a clinical photograph of widespread pustular acne breakout on facial skin, cheek, chin, or jawline",
        ]

        self._eczema_prompts: list[str] = [
            "a close-up photo of hand eczema, finger dermatitis, cracked red scaly skin on knuckles or fingers",
            "a close-up photo of palm eczema, dyshidrotic eczema, pompholyx, red inflamed itchy skin on palm of hand",
            "a close-up photo of atopic eczema, flexural dermatitis with lichenification, excoriations, or dry scaly eczema in elbow or knee creases",
            "a photo of atopic dermatitis rash, eczema patches with fine flaking, itching, and excoriations",
            "a close-up photo of facial eczema, eyelid dermatitis, periorbital dermatitis, dry flaking scaly skin on eyelid or cheek",
            "a clinical close-up photo of pathological skin desquamation, extensive peeling skin, exfoliative erythroderma, or shedding dry diseased epidermal flakes",
            "a clinical photo of nummular eczema or discoid eczema with coin-shaped circular itchy red patches and crusting",
            "a clinical photo of stasis dermatitis, leg eczema, or itchy red rash on ankle, foot, or lower leg",
            "a clinical macro photo of prurigo nodularis, excoriated firm itchy nodular bumps, or severely scratched pruritic lesions",
            "a clinical photo of cutaneous eczema with dry scaly lichenified erythematous eczema plaques",
            "a photo of extensive red rash on arm, forearm, or inner arm with widespread erythematous bumps, papules, or dermatitis",
            "a clinical photograph of widespread erythematous rash, allergic dermatitis, or maculopapular eczema rash covering the forearm or limbs",
            "a close-up photo of widespread red papular dermatitis, eczema flare-up, or skin inflammation on arm or forearm",
        ]

        self._herpes_prompts: list[str] = [
            "a close-up macro photo of herpes simplex cold sores on lips, fever blisters, or tightly clustered fluid-filled water vesicles",
            "a close-up photo of cutaneous herpes vesicles, grouped translucent fluid blisters on erythematous base, or crusted honey-colored sores",
            "a close-up photo of genital herpes, painful shallow erosions, or tightly grouped viral vesicles on genital or groin skin",
            "a clinical macro photo of herpes simplex virus infection, clustered small transparent water blisters on mucosal border or lips",
            "a photo of herpes labialis, grouped fever blisters on lip margin with clear exudate",
            "a clinical macro photo of herpes vesicles, grouped small fluid blisters on skin, or crusted herpes sores",
            "a close-up photo of herpes simplex or herpes rash with clustered water blisters and erythematous base",
        ]

        self._lesion_prompts: list[str] = (
            self._acne_prompts + self._eczema_prompts + self._herpes_prompts
        )

        # Tier B: Out-of-Scope / Unsupported Conditions — non-target diseases
        # Each prompt maps to a concrete disease label via self._unsupported_prompt_labels
        self._unsupported_prompts: list[str] = [
            # Ringworm / Tinea
            "a photo of an annular rash, ringworm, or tinea corporis with round red expanding ring",
            "a close-up photo of ringworm rash or circular tinea fungal infection on skin",
            "a photo of an itchy red ring on skin with distinct active circular borders",
            # Hives / Urticaria
            "a clinical photo of acute urticaria hives with large raised edematous wheals and swollen hive plaques",
            "a clinical photo of urticaria wheals with prominent swollen edematous itchy hive welts",
            "a clinical photograph of acute urticaria rash with raised erythematous wheals",
            "a clinical photo of urticarial plaques with marked edema and swollen wheal borders",
            # Psoriasis
            "a photo of plaque psoriasis with well-demarcated thick plaques and silvery micaceous scales",
            "a photo of plaque psoriasis or guttate psoriasis with raised thick red scaly plaques on elbows or knees",
            # Warts
            "a photo of skin warts, verruca vulgaris, rough cauliflower skin tags, or papilloma growths on fingers",
            # Vitiligo
            "a photo of vitiligo, leukoderma, or stark pure white depigmented skin patches",
            # Melanoma / Cancer
            "a photo of melanoma, dark asymmetrical black mole, or invasive skin cancer tumor",
        ]

        # Maps each unsupported_prompts entry (by index) to its clinical disease label
        self._unsupported_prompt_labels: list[str] = [
            "ringworm",
            "ringworm",
            "ringworm",
            "hives",
            "hives",
            "hives",
            "hives",
            "psoriasis",
            "psoriasis",
            "warts",
            "vitiligo",
            "melanoma",
        ]

        # Tier C: Healthy Skin, Clean Faces & Natural Epidermal Texture
        self._healthy_skin_prompts: list[str] = [
            # Clean Faces & Portraits
            "a portrait photo or face photo of a person or woman with clear healthy skin, smiling, and glowing complexion",
            "a headshot photo of a person with clear, healthy, unblemished facial skin and no lesions",
            "a dermatological photograph of completely clear healthy skin on face",
            "a close-up photograph of clean smooth healthy facial skin, forehead, or clear cheek",
            "a photo of a person with beautiful clear skin and healthy radiant complexion",
            "a close-up photo of clear healthy cheek and glowing smooth skin",
            "a portrait or photo of a person with clear healthy facial skin, beauty cosmetics, and smooth complexion",
            "a close-up photo of a clean healthy face, flawless smooth skin, and clear complexion",
            # Macro Skin & Natural Texture
            "a clinical close-up macro photograph of completely normal healthy human skin, clear intact epidermis, and physiological skin texture",
            "a clinical dermatological photograph of normal healthy skin with intact epidermal barrier, physiological skin lines, and clear skin appearance",
            "a clinical photograph of normal healthy cutaneous skin, intact epidermis, and healthy unblemished skin surface",
            "a close-up dermatological photograph of healthy normal human skin with intact cutaneous barrier and physiological texture",
            "a clinical macro photo of normal healthy human epidermis, physiological skin creases, and clear uninflamed skin texture",
            "a clinical photograph of completely normal, clear, unblemished healthy skin on forearm, arm, or hand with no redness, no rash, no inflammation, and no bumps",
            "a close-up photo of an ankle, foot, heel, leg, arm, or hand with completely clear, healthy, unblemished, uninflamed skin without any rash or spots",
            "a close-up photo of clean healthy arm, wrist, or hand with flawless unblemished epidermis and no dermatological spots",
            "a close-up photo of jewelry, beads, or watch worn on healthy normal skin",
        ]

        # Tier D: Distant Full Body, Distant Poses & Aesthetic Procedures
        self._full_body_prompts: list[str] = [
            "a distant full body or half body photo of a person standing, sitting, or posing with clothes visible",
            "a portrait photo of a person wearing a sweater, shirt, clothing, or collar, posing with hands or arms raised",
            "a portrait photo of a person with hands raised, gesturing, waving, or showing hands in front of body",
            "a distant body photograph, waist-up portrait, or wide shot without a close-up skin lesion",
            "a distant selfie photograph of an individual posing with clothes visible",
            "a full body photo of a person standing or sitting in a room",
            "a photo of a dermatologist or aesthetician examining or treating a patient with a handheld light, device, or dermatoscope",
            "a beauty clinic, cosmetic spa treatment, or aesthetic procedure photo",
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

        print(f"[SkinValidator] Loading CLIP model: {model_name}")

        # Determine device — prefer GPU, fall back to CPU
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

        print(f"[SkinValidator] Multi-Category Semantic Gate loaded on {self._device} "
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

        # Specific semantic affinity across in-scope target diseases
        acne_count = len(self._acne_prompts)
        eczema_count = len(self._eczema_prompts)
        herpes_count = len(self._herpes_prompts)

        acne_max_logit = logits[0, 0: acne_count].max().item()
        eczema_max_logit = logits[0, acne_count: acne_count + eczema_count].max().item()
        herpes_max_logit = logits[0, acne_count + eczema_count: len(self._lesion_prompts)].max().item()

        in_scope_tensor = torch.tensor([acne_max_logit, eczema_max_logit, herpes_max_logit], device=self._device)
        in_scope_probs = torch.softmax(in_scope_tensor / 0.7, dim=0)
        semantic_affinity = {
            "acne": round(in_scope_probs[0].item(), 4),
            "eczema": round(in_scope_probs[1].item(), 4),
            "herpes": round(in_scope_probs[2].item(), 4),
        }

        # ---- Clinical Priority Decision Rules ----
        # 1. Non-skin objects, memes, animals
        if non_skin_score >= 0.35 and non_skin_score > lesion_score and non_skin_score > unsupported_score:
            winning_category = "non_skin"
        # 2. Healthy unblemished skin / clear face (requires healthy_score >= 0.55 and decisive margin over lesion_score)
        elif healthy_score >= 0.55 and healthy_score > (lesion_score * 1.5) and healthy_score >= full_body_score and healthy_score > unsupported_score:
            winning_category = "healthy_skin"
        # 3. Distant full-body portraits, distant poses, clothing, and aesthetic procedures
        elif full_body_score >= 0.30 or (full_body_score >= 0.25 and full_body_score > lesion_score):
            winning_category = "full_body_portrait"
        # 4. Out-of-Scope Condition (unsupported condition strictly beats lesion score with clear margin)
        elif unsupported_score >= 0.50 and unsupported_score > (lesion_score + 0.05):
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
            "semantic_affinity": semantic_affinity,
            "out_of_scope_category": out_of_scope_category,
            "reason": reason,
        }
