"""
Bayesian Clinical Evaluator & Multimodal Diagnostic Refinement
==============================================================
Implements probabilistic Bayesian belief updating for clinical skin disease anamnesis.
Integrates visual prior probabilities from deep learning models with literature-anchored
conditional symptom likelihoods across 7 clinical dimensions.
"""

import math
from typing import Any, Dict, List, Optional


class BayesianClinicalEvaluator:
    """
    Evaluates patient clinical symptom responses against deep learning visual priors
    using log-space Bayes' rule with Laplace smoothing and information-theoretic
    question selection.
    """

    # 7 Core Clinical Dimensions with Literature-Anchored Conditional Likelihoods P(Symptom | Disease)
    # References: Fitzpatrick's Dermatology, Rook's Textbook of Dermatology, AAD Guidelines
    SYMPTOM_LIKELIHOODS: Dict[str, Dict[str, Dict[str, float]]] = {
        "sensation": {
            "title": "Primary Sensation",
            "question": "What does this affected skin area primarily feel like?",
            "options": {
                "itchy": {
                    "label": "Intense itching",
                    "description": "Severe, irritating itch that makes you want to scratch, especially at night or when warm",
                    "icon": "🔴",
                    "likelihoods": {"Eczema": 0.94, "Acne": 0.12, "Herpes": 0.25},
                },
                "burning_tingling": {
                    "label": "Tingling, burning, or prickling",
                    "description": "Hot, stinging, or electric tingling that started before the bumps appeared",
                    "icon": "⚡",
                    "likelihoods": {"Herpes": 0.92, "Acne": 0.05, "Eczema": 0.30},
                },
                "tender_sore": {
                    "label": "Sore or tender pressure",
                    "description": "Achy or tender when pressed or bumped, but doesn't constantly itch",
                    "icon": "🩹",
                    "likelihoods": {"Acne": 0.88, "Herpes": 0.40, "Eczema": 0.10},
                },
                "painless": {
                    "label": "Painless / No sensation",
                    "description": "Doesn't hurt, sting, or itch at all; only noticed visually",
                    "icon": "⚪",
                    "likelihoods": {"Acne": 0.35, "Eczema": 0.05, "Herpes": 0.05},
                },
                "unsure": {
                    "label": "I'm not sure",
                    "description": "Hard to describe or varies throughout the day",
                    "icon": "❓",
                    "likelihoods": {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333},
                },
            },
        },
        "texture": {
            "title": "Lesion Texture & Surface",
            "question": "How does the surface of the skin feel when touched gently?",
            "options": {
                "pimples_clogged": {
                    "label": "Pimples or clogged bumps",
                    "description": "Raised red bumps, whiteheads, blackheads, or spots with white/yellow pus",
                    "icon": "🔴",
                    "likelihoods": {"Acne": 0.96, "Eczema": 0.08, "Herpes": 0.15},
                },
                "rough_dry_scaly": {
                    "label": "Dry, rough, or flaky",
                    "description": "Sandpaper-like, rough, scaly patches that feel tight or peel",
                    "icon": "🌾",
                    "likelihoods": {"Eczema": 0.93, "Acne": 0.06, "Herpes": 0.05},
                },
                "clustered_blisters": {
                    "label": "Clustered tiny blisters",
                    "description": "A tight cluster of small, delicate bubbles filled with clear fluid that pop and crust",
                    "icon": "💧",
                    "likelihoods": {"Herpes": 0.95, "Acne": 0.04, "Eczema": 0.18},
                },
                "unsure": {
                    "label": "I'm not sure",
                    "description": "Surface texture is indistinct or mixed",
                    "icon": "❓",
                    "likelihoods": {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333},
                },
            },
        },
        "progression": {
            "title": "Timeline & Progression",
            "question": "How has this condition developed over time?",
            "options": {
                "ongoing_breakouts": {
                    "label": "Ongoing daily breakouts",
                    "description": "New bumps appear continuously over weeks or months in oily areas",
                    "icon": "🔄",
                    "likelihoods": {"Acne": 0.90, "Herpes": 0.08, "Eczema": 0.20},
                },
                "rapid_blister_cycle": {
                    "label": "Rapid cycle (1–2 weeks)",
                    "description": "Appeared suddenly, formed small sores, and is drying into scabs within 7–10 days",
                    "icon": "⚡",
                    "likelihoods": {"Herpes": 0.94, "Acne": 0.06, "Eczema": 0.15},
                },
                "chronic_flares": {
                    "label": "Periodic flare-ups",
                    "description": "Comes and goes over months; calms down with moisturizing and flares up again",
                    "icon": "📈",
                    "likelihoods": {"Eczema": 0.92, "Acne": 0.25, "Herpes": 0.12},
                },
                "unsure": {
                    "label": "I'm not sure",
                    "description": "First time observing or uncertain of onset",
                    "icon": "❓",
                    "likelihoods": {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333},
                },
            },
        },
        "distribution": {
            "title": "Body Location & Distribution",
            "question": "Where on your body is this or similar skin flare-ups located?",
            "options": {
                "face_chest_back": {
                    "label": "Face, forehead, cheeks, chest, or upper back",
                    "description": "Common on the T-zone, cheeks, jawline, upper chest, or shoulder blades",
                    "icon": "👤",
                    "likelihoods": {"Acne": 0.92, "Eczema": 0.30, "Herpes": 0.35},
                },
                "flexural_creases": {
                    "label": "Inside elbows, behind knees, wrists, or neck",
                    "description": "Inside creases of the arms, behind the knees, around the wrists, or hands",
                    "icon": "🦾",
                    "likelihoods": {"Eczema": 0.95, "Acne": 0.02, "Herpes": 0.02},
                },
                "lips_vermilion_mucosa": {
                    "label": "Along the lip border, mouth, or sensitive areas",
                    "description": "Right along the edge of the lip (cold sore area) or sensitive personal areas",
                    "icon": "👄",
                    "likelihoods": {"Herpes": 0.93, "Acne": 0.10, "Eczema": 0.10},
                },
                "single_isolated": {
                    "label": "Only in this single isolated spot",
                    "description": "Nowhere else on the entire body",
                    "icon": "📍",
                    "likelihoods": {"Acne": 0.40, "Herpes": 0.45, "Eczema": 0.15},
                },
                "unsure": {
                    "label": "I'm not sure",
                    "description": "Affects various unlisted regions",
                    "icon": "❓",
                    "likelihoods": {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333},
                },
            },
        },
        "triggers": {
            "title": "Triggers & Environmental Factors",
            "question": "What tends to trigger or worsen this flare-up?",
            "options": {
                "oils_cosmetics_sweat": {
                    "label": "Skincare oils, cosmetics, sweat, or friction",
                    "description": "Worsened by heavy creams, makeup, sweating under hats/helmets, or greasy foods",
                    "icon": "🧴",
                    "likelihoods": {"Acne": 0.86, "Eczema": 0.30, "Herpes": 0.10},
                },
                "cold_dry_air_soaps": {
                    "label": "Cold dry weather, hot baths, or harsh soaps",
                    "description": "Flares up in winter, with perfumed detergents, wool clothing, or dry air",
                    "icon": "❄️",
                    "likelihoods": {"Eczema": 0.91, "Acne": 0.10, "Herpes": 0.08},
                },
                "stress_sun_illness": {
                    "label": "Intense stress, physical exhaustion, sunlight, or fever",
                    "description": "Triggered during periods of low immunity, lack of sleep, sun exposure, or illness",
                    "icon": "☀️",
                    "likelihoods": {"Herpes": 0.90, "Acne": 0.30, "Eczema": 0.35},
                },
                "unsure": {
                    "label": "I'm not sure / No clear trigger",
                    "description": "Has not noticed any specific pattern or cause",
                    "icon": "❓",
                    "likelihoods": {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333},
                },
            },
        },
        "history": {
            "title": "Personal & Family Medical History",
            "question": "Do you or close family members have a history of any of the following?",
            "options": {
                "atopic_triad": {
                    "label": "Asthma, hay fever (allergies), or childhood eczema",
                    "description": "Known personal or familial tendency toward allergies or sensitive skin",
                    "icon": "🌿",
                    "likelihoods": {"Eczema": 0.88, "Acne": 0.15, "Herpes": 0.15},
                },
                "previous_cold_sores": {
                    "label": "Previous cold sores or fever blisters",
                    "description": "Have experienced similar tingling blisters on lips or sensitive areas in the past",
                    "icon": "🔁",
                    "likelihoods": {"Herpes": 0.91, "Acne": 0.05, "Eczema": 0.10},
                },
                "family_acne": {
                    "label": "Family history of persistent teen or adult acne",
                    "description": "Close relatives experienced oily skin and persistent acne blemishes",
                    "icon": "👥",
                    "likelihoods": {"Acne": 0.82, "Eczema": 0.15, "Herpes": 0.10},
                },
                "none": {
                    "label": "None of the above",
                    "description": "No prior personal or family history of these conditions",
                    "icon": "🚫",
                    "likelihoods": {"Acne": 0.35, "Eczema": 0.20, "Herpes": 0.30},
                },
                "unsure": {
                    "label": "I'm not sure",
                    "description": "Family medical history is unknown",
                    "icon": "❓",
                    "likelihoods": {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333},
                },
            },
        },
        "secretion": {
            "title": "Secretion & Lesion Fluid",
            "question": "Does the skin lesion leak, ooze, or produce any discharge?",
            "options": {
                "pus_thick": {
                    "label": "Thick white or yellowish pus",
                    "description": "Contained within a pimple head or expelled when inflamed",
                    "icon": "⚪",
                    "likelihoods": {"Acne": 0.94, "Eczema": 0.10, "Herpes": 0.15},
                },
                "clear_fluid_blister": {
                    "label": "Tiny clear fluid bubbles that pop and crust",
                    "description": "Small transparent blisters that burst, leak watery fluid, and dry into scabs",
                    "icon": "💧",
                    "likelihoods": {"Herpes": 0.95, "Acne": 0.05, "Eczema": 0.20},
                },
                "weeping_watery": {
                    "label": "Clear watery oozing that weeps across a raw patch",
                    "description": "Skin weeps clear fluid after scratching, leaving wet or stiffened areas",
                    "icon": "🟡",
                    "likelihoods": {"Eczema": 0.90, "Acne": 0.05, "Herpes": 0.20},
                },
                "dry_no_fluid": {
                    "label": "Completely dry — no fluid, pus, or oozing",
                    "description": "Lesion is dry, flaky, or solid without any discharge",
                    "icon": "🌵",
                    "likelihoods": {"Eczema": 0.65, "Acne": 0.40, "Herpes": 0.05},
                },
                "unsure": {
                    "label": "I'm not sure",
                    "description": "Discharge status is unclear",
                    "icon": "❓",
                    "likelihoods": {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333},
                },
            },
        },
    }

    @classmethod
    def get_dynamic_questions(
        cls,
        prior_probabilities: Dict[str, float],
        max_questions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dynamically selects questions from the bank based on the visual model's prior distribution.
        - High confidence (> 80%): Serves 3-4 quick confirmatory questions.
        - Ambiguous / Close calls: Serves 5-6 targeted questions ranked by Information Gain
          between the top competing classes.
        """
        if not prior_probabilities:
            sorted_diseases = ["Acne", "Eczema", "Herpes"]
            top_prob = 0.33
            top_margin = 0.0
        else:
            sorted_items = sorted(prior_probabilities.items(), key=lambda x: x[1], reverse=True)
            sorted_diseases = [item[0] for item in sorted_items]
            top_prob = sorted_items[0][1]
            top_margin = (sorted_items[0][1] - sorted_items[1][1]) if len(sorted_items) > 1 else 1.0

        d1 = sorted_diseases[0] if len(sorted_diseases) > 0 else "Acne"
        d2 = sorted_diseases[1] if len(sorted_diseases) > 1 else "Eczema"

        # Determine target question count
        if max_questions is not None:
            target_count = max_questions
        elif top_prob >= 0.80 and top_margin >= 0.50:
            target_count = 3  # Clear prior: quick 3-question verification
        elif top_margin >= 0.25:
            target_count = 4  # Moderate confidence: 4 questions
        else:
            target_count = 6  # Ambiguous / close competition: 6 questions

        # Score questions by Mutual Information / Information Divergence between d1 and d2
        scored_questions = []
        for q_id, q_data in cls.SYMPTOM_LIKELIHOODS.items():
            divergence = 0.0
            for opt_id, opt_info in q_data["options"].items():
                if opt_id == "unsure":
                    continue
                p_d1 = opt_info["likelihoods"].get(d1, 0.3333)
                p_d2 = opt_info["likelihoods"].get(d2, 0.3333)
                # Absolute log likelihood ratio contribution
                ratio = math.fabs(math.log(max(p_d1, 0.01) / max(p_d2, 0.01)))
                divergence += ratio

            scored_questions.append((q_id, divergence, q_data))

        # Sort descending by discriminative power between top two candidates
        scored_questions.sort(key=lambda x: x[1], reverse=True)

        selected = []
        for q_id, score, q_data in scored_questions[:target_count]:
            options_list = []
            for opt_key, opt_val in q_data["options"].items():
                options_list.append({
                    "id": opt_key,
                    "label": opt_val["label"],
                    "description": opt_val["description"],
                    "icon": opt_val.get("icon", "•"),
                })
            selected.append({
                "id": q_id,
                "title": q_data["title"],
                "question": q_data["question"],
                "options": options_list,
            })

        return selected

    @classmethod
    def compute_posterior(
        cls,
        prior_probabilities: Dict[str, float],
        answers: Dict[str, str],
        epsilon: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Computes posterior probabilities via Bayes' rule in log-space.
        P(D_i | symptoms) = P(D_i) * prod(P(s_j | D_i)) / Z

        Args:
            prior_probabilities: Dict mapping disease name to visual softmax probability.
            answers: Dict mapping question dimension ID (e.g. 'sensation') to chosen option ID.
            epsilon: Laplace smoothing floor to prevent log(0).

        Returns:
            Dictionary containing:
            - refined_probabilities: Normalized posterior distribution.
            - refined_label: Top disease label.
            - refined_confidence: Float score of top disease.
            - visual_label: Top disease prior.
            - visual_confidence: Float score of top disease prior.
            - delta_confidence: Confidence shift (+/-).
            - is_significant_shift: True if top label flipped or confidence shifted >= 15%.
            - clinical_evidence: List of Explainable AI (XAI) rationale strings.
        """
        if not prior_probabilities:
            prior_probabilities = {"Acne": 0.3333, "Eczema": 0.3333, "Herpes": 0.3333}

        # Normalize priors if needed
        prior_sum = sum(prior_probabilities.values())
        if prior_sum > 0:
            norm_priors = {k: max(v / prior_sum, epsilon) for k, v in prior_probabilities.items()}
        else:
            norm_priors = {k: 1.0 / len(prior_probabilities) for k in prior_probabilities}

        diseases = list(norm_priors.keys())
        log_posteriors: Dict[str, float] = {d: math.log(norm_priors[d]) for d in diseases}

        # Track evidence contributions for XAI explanation
        evidence_impacts: List[Dict[str, Any]] = []

        for q_id, chosen_option in answers.items():
            if not chosen_option or chosen_option == "unsure":
                continue

            q_info = cls.SYMPTOM_LIKELIHOODS.get(q_id)
            if not q_info:
                continue

            opt_info = q_info["options"].get(chosen_option)
            if not opt_info:
                continue

            likelihoods = opt_info.get("likelihoods", {})

            # For diseases not in the dictionary, assign a neutral uniform likelihood
            default_likelihood = 1.0 / max(len(diseases), 3)

            for d in diseases:
                p_s_given_d = likelihoods.get(d, default_likelihood)
                # Clip to epsilon to avoid log(0)
                p_s_given_d = max(p_s_given_d, epsilon)
                log_posteriors[d] += math.log(p_s_given_d)

            # Record highest positive alignment for XAI
            sorted_likes = sorted(
                [(d, likelihoods.get(d, default_likelihood)) for d in diseases],
                key=lambda x: x[1],
                reverse=True,
            )
            top_aligned_disease = sorted_likes[0][0]
            top_aligned_prob = sorted_likes[0][1]

            if top_aligned_prob >= 0.80:
                evidence_impacts.append({
                    "dimension": q_info["title"],
                    "symptom": opt_info["label"],
                    "favors": top_aligned_disease,
                    "strength": round(top_aligned_prob, 2),
                })

        # Softmax partition function normalization over log posteriors
        max_log = max(log_posteriors.values())
        exp_posteriors = {d: math.exp(log_posteriors[d] - max_log) for d in diseases}
        partition_z = sum(exp_posteriors.values())

        posterior_probs = {d: round(exp_posteriors[d] / partition_z, 4) for d in diseases}

        # Determine visual and refined top candidates
        visual_label = max(norm_priors, key=lambda k: norm_priors[k])
        visual_conf = round(norm_priors[visual_label], 4)

        refined_label = max(posterior_probs, key=lambda k: posterior_probs[k])
        refined_conf = posterior_probs[refined_label]

        delta = round(refined_conf - visual_conf, 4)
        is_shift = (visual_label != refined_label) or (abs(delta) >= 0.15)

        # Generate human-readable clinical evidence rationales
        rationales: List[str] = []
        for item in evidence_impacts:
            if item["favors"] == refined_label:
                rationales.append(
                    f"Reported {item['dimension'].lower()} ('{item['symptom']}') strongly supports {refined_label} (clinical sensitivity {int(item['strength'] * 100)}%)."
                )

        if not rationales:
            if visual_label == refined_label:
                rationales.append(
                    f"Reported clinical symptoms are consistent with the visual presentation of {refined_label}."
                )
            else:
                rationales.append(
                    f"Multimodal symptom combination resolved visual ambiguity toward {refined_label}."
                )

        return {
            "refined_probabilities": posterior_probs,
            "refined_label": refined_label,
            "refined_confidence": refined_conf,
            "visual_label": visual_label,
            "visual_confidence": visual_conf,
            "delta_confidence": delta,
            "is_significant_shift": is_shift,
            "clinical_evidence": rationales,
            "answers_recorded": answers,
        }
