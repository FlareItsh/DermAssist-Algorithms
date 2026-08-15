"""
!!! SYSTEM FLARE: API/APP.PY IS STARTING NOW !!!
================================================
"""
import os
import sys

# Force UTF-8 encoding on standard streams to prevent Windows console UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import yaml
import torch
import uvicorn
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from src.data_loader import load_config
from src.inference import SkinLesionPredictor, EnsemblePredictor
from src.skin_validator import SkinValidator
from src.image_quality import ImageQualityAssessor

# ============================================================
# Schemas
# ============================================================

class Prediction(BaseModel):
    label: str
    confidence: float

class ImageQuality(BaseModel):
    is_blurry: bool
    is_dark: bool
    is_overexposed: bool
    is_low_contrast: bool
    sharpness_score: float
    brightness_score: float
    contrast_score: float
    status: str
    feedback_message: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    all_probabilities: dict
    device: str
    architecture: str
    image_type: str
    image_quality: Optional[ImageQuality] = None

class ValidationResponse(BaseModel):
    is_skin: bool
    skin_score: float
    reason: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

class ClassesResponse(BaseModel):
    classes: List[str]
    num_classes: int


# ---- Global Model Loading (Eager) ----
print("\n" + "="*60)
print(" 🚀 DERMASSIST AI: EAGER LOADING STARTING...")
print("="*60)

def init_predictor_sync():
    """Synchronously load the model at top-level."""
    print(f"[DEBUG] Current Directory: {os.getcwd()}")
    try:
        # Load settings from config
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            print(f"[API] ❌ ERROR: Cannot find {config_path}")
            return None
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        active_arch = config["inference"].get("active_inference_model", "resnet50")
        
        # Use config path, fallback to legacy if specifically requested
        model_path = config["training"].get("production_model_path", os.path.join("models/production", "best_model.pth"))
        
        ensemble_enabled = config["inference"].get("ensemble_enabled", False)
        
        if ensemble_enabled:
            print(f"[API] ⏳ LOADING ENSEMBLE MODEL")
            model_paths_and_archs = [
                ("models/production/best_model_resnet50.pth", "resnet50"),
                ("models/production/best_model_efficientnet_v2.pth", "efficientnet_v2"),
                ("models/production/best_model_swin_transformer.pth", "swin_transformer"),
            ]
            predictor = EnsemblePredictor(
                model_paths_and_archs=model_paths_and_archs,
                config=config,
                device="cpu",
            )
        else:
            # Check if model exists
            if not os.path.exists(model_path):
                legacy_path = os.path.join("models/production", "best_3class_legacy.pth")
                if os.path.exists(legacy_path):
                    model_path = legacy_path
                else:
                    print(f"[API] ⚠ WARNING: No model found at {model_path}. Predict endpoint will be disabled.")
                    return None

            print(f"[API] ⏳ LOADING MODEL: {active_arch.upper()} from {model_path}")
            
            predictor = SkinLesionPredictor(
                model_path=model_path,
                config=config,
                device="cpu", 
                architecture=active_arch
            )
        
        print(f"============================================================")
        print(f" 🎉 SUCCESS: AI BRAIN READY!")
        print(f"============================================================\n")
        return predictor
        
    except Exception as e:
        import traceback
        print(f"\n[API] ❌ CRITICAL EAGER LOAD FAILURE:")
        print(traceback.format_exc())
        return None

# Load it NOW
predictor = init_predictor_sync()

# ---- Load CLIP skin validator ----
print("[API] ⏳ Loading skin validator...")
try:
    with open("config.yaml", "r") as _f:
        _cfg = yaml.safe_load(_f)
    skin_validator = SkinValidator(_cfg)
except Exception as _e:
    print(f"[API] ⚠ Skin validator failed to load: {_e}")
    skin_validator = None

# ---- Image quality assessor (observational only — never alters images) ----
quality_assessor = ImageQualityAssessor()

# ============================================================
# Application Setup
# ============================================================

app = FastAPI(
    title="DermAssist AI API",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "N/A",
    )

@app.get("/classes", response_model=ClassesResponse)
async def get_classes():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ClassesResponse(
        classes=predictor.class_names,
        num_classes=len(predictor.class_names),
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="AI Brain is not loaded.")

    try:
        # Save temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # ---- Skin validation gate ----
        image_type = "unknown"
        if skin_validator is not None:
            validation = skin_validator.validate(temp_path)
            image_type = validation.get("image_type", "unknown")
            if not validation["is_skin"]:
                # Transient image quality check for the non-skin image
                quality_result = quality_assessor.assess(temp_path)
                os.remove(temp_path)
                
                image_quality = None
                if quality_result is not None:
                    image_quality = ImageQuality(**quality_result.to_dict())
                
                return PredictionResponse(
                    label="None",
                    confidence=1.0,
                    all_probabilities={"Acne": 0.0, "Eczema": 0.0, "Herpes": 0.0},
                    device="cpu",
                    architecture="clip",
                    image_type=image_type,
                    image_quality=image_quality,
                )


        # ---- Image quality assessment (metadata only — image unchanged) ----
        # Runs on the same temp file BEFORE the predictor so we can clean up
        # after both calls. This never modifies the file or affects inference.
        quality_result = quality_assessor.assess(temp_path)

        # Inference
        result = predictor.predict(temp_path)

        # Cleanup
        os.remove(temp_path)

        image_quality = None
        if quality_result is not None:
            image_quality = ImageQuality(**quality_result.to_dict())

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            device=str(predictor.device),
            architecture=predictor.architecture,
            image_type=image_type,
            image_quality=image_quality,
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("\n" + "!"*60)
        print(" ❌ PREDICTION CRASH:")
        print(traceback.format_exc())
        print("!"*60 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Load port from config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            port = config["api"].get("port", 8001)
    except:
        port = 8001

    print(f"[API] Starting server on port {port}...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=True)