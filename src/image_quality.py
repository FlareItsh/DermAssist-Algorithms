"""
DermAssist - Image Quality Assessor
=====================================
Evaluates the quality of an uploaded skin scan image using OpenCV.

This module is PURELY OBSERVATIONAL — it reads the image file independently
and never modifies, compresses, or re-encodes it. The original image bytes
that are passed to the disease predictor remain completely untouched.

Metrics:
    - Sharpness  : Laplacian variance on grayscale image
    - Brightness : Mean pixel intensity of grayscale image
    - Contrast   : Standard deviation of pixel intensities

Usage:
    from src.image_quality import ImageQualityAssessor, ImageQualityResult

    assessor = ImageQualityAssessor()
    result = assessor.assess(image_path)
    print(result.status, result.feedback_message)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# Lazy OpenCV import — don't crash the API if cv2 is absent
# ============================================================
try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ============================================================
# Thresholds
# ============================================================

# Sharpness: Laplacian variance. Lower = blurrier.
# Research-backed range for macro photography: >100 is very sharp, <50 is blurry.
# We use 80 as a conservative cutoff that flags genuinely blurry images.
SHARPNESS_THRESHOLD = 80.0

# Brightness: Mean pixel intensity (0–255).
# < 55  → Too dark (underexposed; lesion details invisible)
# > 215 → Overexposed (glare washes out texture)
BRIGHTNESS_DARK_THRESHOLD = 55.0
BRIGHTNESS_OVEREXPOSED_THRESHOLD = 215.0

# Contrast: Std deviation of pixel intensities.
# A well-exposed skin lesion photo should have texture variation.
# < 22 = near-uniform image (e.g. pure white/black or extreme glare)
CONTRAST_LOW_THRESHOLD = 22.0


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class ImageQualityResult:
    """
    Scan-time image quality metadata.

    All boolean flags express a quality problem (True = problem present).
    This result is appended to PredictionResponse as metadata only.

    Attributes:
        is_blurry (bool): Laplacian variance below threshold.
        is_dark (bool): Mean brightness below dark threshold.
        is_overexposed (bool): Mean brightness above glare threshold.
        is_low_contrast (bool): Std deviation too low for reliable analysis.
        sharpness_score (float): Raw Laplacian variance (higher = sharper).
        brightness_score (float): Raw mean pixel intensity (0–255).
        contrast_score (float): Raw pixel std deviation.
        status (str): Human-readable summary label.
        feedback_message (str): Patient-facing explanation.
    """

    is_blurry: bool
    is_dark: bool
    is_overexposed: bool
    is_low_contrast: bool
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    status: str
    feedback_message: str

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON responses."""
        return {
            "is_blurry": self.is_blurry,
            "is_dark": self.is_dark,
            "is_overexposed": self.is_overexposed,
            "is_low_contrast": self.is_low_contrast,
            "sharpness_score": round(self.sharpness_score, 2),
            "brightness_score": round(self.brightness_score, 2),
            "contrast_score": round(self.contrast_score, 2),
            "status": self.status,
            "feedback_message": self.feedback_message,
        }


# ============================================================
# Assessor class
# ============================================================

class ImageQualityAssessor:
    """
    Stateless image quality assessor.

    Reads the image file from disk using OpenCV, computes quality
    metrics on the grayscale representation, and returns an
    ``ImageQualityResult`` with human-readable status flags.

    The original file is NEVER modified.
    """

    def assess(self, image_path: str) -> Optional[ImageQualityResult]:
        """
        Assess the quality of an image file.

        Args:
            image_path: Absolute path to the image file on disk.

        Returns:
            ``ImageQualityResult`` if assessment succeeds, or ``None``
            if OpenCV is unavailable or the file cannot be read.
        """
        if not _CV2_AVAILABLE:
            return None

        try:
            # ── Load as colour then convert to grayscale ──────────────────
            # cv2.IMREAD_COLOR ensures consistent 3-channel load.
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                return None

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # ── Sharpness via Laplacian variance ──────────────────────────
            # Apply a Laplacian kernel and measure the variance of the
            # response. A focused image has strong edges → high variance.
            # A blurry image has smooth gradients → low variance.
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # ── Brightness via mean pixel intensity ───────────────────────
            mean_brightness = float(np.mean(gray))

            # ── Contrast via pixel std deviation ─────────────────────────
            contrast = float(np.std(gray))

            # ── Classify issues ───────────────────────────────────────────
            is_blurry = laplacian_var < SHARPNESS_THRESHOLD
            is_dark = mean_brightness < BRIGHTNESS_DARK_THRESHOLD
            is_overexposed = mean_brightness > BRIGHTNESS_OVEREXPOSED_THRESHOLD
            is_low_contrast = contrast < CONTRAST_LOW_THRESHOLD

            # ── Derive status and feedback ────────────────────────────────
            status, feedback_message = self._build_feedback(
                is_blurry, is_dark, is_overexposed, is_low_contrast
            )

            return ImageQualityResult(
                is_blurry=is_blurry,
                is_dark=is_dark,
                is_overexposed=is_overexposed,
                is_low_contrast=is_low_contrast,
                sharpness_score=laplacian_var,
                brightness_score=mean_brightness,
                contrast_score=contrast,
                status=status,
                feedback_message=feedback_message,
            )

        except Exception as exc:
            # Never crash the prediction pipeline — quality check is optional
            print(f"[ImageQualityAssessor] ⚠ Quality check failed silently: {exc}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_feedback(
        self,
        is_blurry: bool,
        is_dark: bool,
        is_overexposed: bool,
        is_low_contrast: bool,
    ) -> tuple[str, str]:
        """
        Return a (status, feedback_message) pair based on the detected issues.

        Priority order: dark > overexposed > blurry > low contrast > excellent.
        If multiple issues exist, the most impactful is reported.
        """
        issues: list[str] = []
        if is_dark:
            issues.append("too dark")
        if is_overexposed:
            issues.append("overexposed / glare")
        if is_blurry:
            issues.append("blurry")
        if is_low_contrast:
            issues.append("low contrast")

        if not issues:
            return (
                "Excellent",
                "The image is sharp, well-lit, and has good contrast. "
                "The AI accuracy score is highly reliable.",
            )

        if is_dark:
            return (
                "Poor Lighting",
                "The image appears too dark. Skin tone details and lesion features "
                "may not be clearly visible, which can reduce prediction accuracy. "
                "Try scanning in a well-lit area or closer to a light source.",
            )

        if is_overexposed:
            return (
                "Overexposed",
                "The image has too much glare or is overexposed. "
                "This can wash out lesion details and lower prediction accuracy. "
                "Try avoiding direct flash or bright reflective light sources.",
            )

        if is_blurry:
            return (
                "Blurry",
                "The image appears blurry or out of focus. "
                "Sharp, clear edges help the model identify skin features accurately. "
                "Try holding the camera steady and ensuring the lesion is in focus.",
            )

        if is_low_contrast:
            return (
                "Low Contrast",
                "The image has very low contrast — the lesion may not stand out "
                "clearly from the surrounding skin. Try adjusting lighting so the "
                "affected area is more visible.",
            )

        # Fallback (should not be reached)
        return (
            "Acceptable",
            "The image quality is acceptable but could be improved for best results.",
        )
