"""
Unit tests for Bayesian Clinical Evaluator (using standard library unittest)
"""
import unittest
from src.bayesian_evaluator import BayesianClinicalEvaluator


class TestBayesianClinicalEvaluator(unittest.TestCase):
    def test_neutral_answers_preserve_prior(self):
        priors = {"Acne": 0.60, "Eczema": 0.25, "Herpes": 0.15}
        answers = {
            "sensation": "unsure",
            "texture": "unsure",
            "progression": "unsure",
        }
        result = BayesianClinicalEvaluator.compute_posterior(priors, answers)
        self.assertEqual(result["refined_label"], "Acne")
        self.assertAlmostEqual(result["refined_probabilities"]["Acne"], 0.60, delta=0.02)
        self.assertAlmostEqual(result["refined_probabilities"]["Eczema"], 0.25, delta=0.02)
        self.assertAlmostEqual(result["refined_probabilities"]["Herpes"], 0.15, delta=0.02)

    def test_symptom_evidence_resolves_ambiguity(self):
        # Ambiguous 50/50 prior between Acne and Herpes
        priors = {"Acne": 0.50, "Herpes": 0.50, "Eczema": 0.0}
        answers = {
            "sensation": "tender_sore",
            "texture": "pimples_clogged",
            "distribution": "face_chest_back",
            "secretion": "pus_thick",
        }
        result = BayesianClinicalEvaluator.compute_posterior(priors, answers)
        self.assertEqual(result["refined_label"], "Acne")
        self.assertGreater(result["refined_confidence"], 0.90)
        self.assertLess(result["refined_probabilities"]["Herpes"], 0.10)
        self.assertTrue(result["is_significant_shift"])
        self.assertGreater(len(result["clinical_evidence"]), 0)

    def test_probabilities_sum_to_one(self):
        priors = {"Acne": 0.40, "Eczema": 0.35, "Herpes": 0.25}
        answers = {
            "sensation": "itchy",
            "texture": "rough_dry_scaly",
        }
        result = BayesianClinicalEvaluator.compute_posterior(priors, answers)
        total = sum(result["refined_probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_dynamic_question_selection_counts(self):
        # High confidence -> 3 questions
        high_conf_priors = {"Acne": 0.92, "Eczema": 0.05, "Herpes": 0.03}
        q_high = BayesianClinicalEvaluator.get_dynamic_questions(high_conf_priors)
        self.assertEqual(len(q_high), 3)

        # Ambiguous -> 6 questions
        ambig_priors = {"Acne": 0.48, "Herpes": 0.46, "Eczema": 0.06}
        q_ambig = BayesianClinicalEvaluator.get_dynamic_questions(ambig_priors)
        self.assertEqual(len(q_ambig), 6)


if __name__ == "__main__":
    unittest.main()
