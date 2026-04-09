"""
DermAssist - FastAPI Application
==================================
REST API for skin lesion classification.

Endpoints:
    POST /predict  — Upload an image and receive classification results.
    GET  /health   — Health check endpoint.
    GET  /classes  — List supported skin lesion classes.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import io
from typing import Dict, List

import yaml
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import load_predictor


# ============================================================
# Response Models
# ============================================================

class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    label: str
    confidence: float
    class_index: int
    all_probabilities: Dict[str, float]


class TopKPrediction(BaseModel):
    """Single prediction entry."""
    label: str
    confidence: float


class TopKResponse(BaseModel):
    """Response schema for top-K predictions."""
    predictions: List[TopKPrediction]


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    model_loaded: bool
    device: str


class ClassesResponse(BaseModel):
    """Response schema for the /classes endpoint."""
    classes: List[str]
    num_classes: int


# ============================================================
# Application Setup
# ============================================================

app = FastAPI(
    title="DermAssist — Skin Lesion Detection API",
    description=(
        "Upload a dermoscopic image and receive an AI-powered "
        "skin lesion classification with confidence scores."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow requests from the DermAssist frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Global predictor (loaded once at startup) ----
predictor = None


@app.on_event("startup")
async def startup_event():
    """Load the ML model once when the server starts."""
    global predictor
    try:
        config_path = os.environ.get("DERMASSIST_CONFIG", "config.yaml")
        model_path = os.environ.get("DERMASSIST_MODEL", None)
        predictor = load_predictor(
            config_path=config_path,
            model_path=model_path,
        )
        print("[API] ✅ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"[API] ⚠ Model file not found: {e}")
        print("[API] The /predict endpoint will return 503 until a model is available.")
    except Exception as e:
        print(f"[API] ⚠ Failed to load model: {e}")
        print("[API] The /predict endpoint will return 503 until a model is available.")


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "N/A",
    )


@app.get("/classes", response_model=ClassesResponse, tags=["System"])
async def get_classes():
    """List all supported skin lesion classes."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )
    return ClassesResponse(
        classes=predictor.class_names,
        num_classes=len(predictor.class_names),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a skin lesion image.

    Accepts a multipart image file (JPEG, PNG, BMP, TIFF).
    Returns the predicted label and confidence score.

    **Example response:**
    ```json
    {
        "label": "Melanoma",
        "confidence": 0.9234,
        "class_index": 4,
        "all_probabilities": {
            "Actinic keratoses": 0.0021,
            "Basal cell carcinoma": 0.0103,
            ...
        }
    }
    ```
    """
    # Validate model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first and place "
                   "the .pth file in models/production/.",
        )

    # Validate file type
    allowed_types = {
        "image/jpeg", "image/png", "image/bmp",
        "image/tiff", "image/webp",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                   f"Allowed: {', '.join(allowed_types)}",
        )

    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        result = predictor.predict(image)

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            class_index=result["class_index"],
            all_probabilities=result["all_probabilities"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/top-k", response_model=TopKResponse, tags=["Prediction"])
async def predict_top_k(file: UploadFile = File(...), k: int = 3):
    """
    Get top-K predictions for a skin lesion image.

    Args:
        file: Multipart image file.
        k:    Number of top predictions to return (default: 3).
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        results = predictor.predict_top_k(image, k=k)

        return TopKResponse(
            predictions=[
                TopKPrediction(label=r["label"], confidence=r["confidence"])
                for r in results
            ]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


# ============================================================
# Run directly with: python api/app.py
# ============================================================

if __name__ == "__main__":
    import uvicorn

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    host = config["api"]["host"]
    port = config["api"]["port"]

    print(f"\n🚀 Starting DermAssist API on http://{host}:{port}")
    print(f"📖 Interactive docs at http://{host}:{port}/docs\n")

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=True,
    )
