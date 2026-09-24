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
from src.training_manager import training_manager

# ============================================================
# Schemas
# ============================================================

class StartTrainingRequest(BaseModel):
    architecture: Optional[str] = "ensemble"
    epochs: Optional[int] = 5
    sync_dataset: Optional[bool] = True
    learning_rate: Optional[float] = None

class TrainingStatusResponse(BaseModel):
    status: str
    progress: float
    architecture: str
    current_epoch: int
    total_epochs: int
    current_batch: int
    total_batches: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    baseline_val_acc: float
    best_val_acc: float
    model_promoted: bool
    message: str
    eta_seconds: int
    elapsed_seconds: int
    logs: List[str]
    history: dict
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

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
    is_inconclusive: bool = False
    clinical_feedback: Optional[str] = None

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

def reload_predictor():
    """Reload predictor with newly trained weights."""
    global predictor
    print("\n[API] 🔄 Hot-reloading AI model predictor into memory...")
    try:
        predictor = init_predictor_sync()
        return predictor is not None
    except Exception as e:
        print(f"[API] ❌ Failed to reload predictor: {e}")
        return False

# Register reload callback with the TrainingManager
training_manager.register_reload_callback(reload_predictor)

# ---- Load CLIP skin validator ----
print("[API] ⏳ Loading skin validator...")
try:
    import importlib
    import src.skin_validator
    importlib.reload(src.skin_validator)
    with open("config.yaml", "r") as _f:
        _cfg = yaml.safe_load(_f)
    skin_validator = src.skin_validator.SkinValidator(_cfg)
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

        # ---- Semantic multi-tier clinical gate ----
        global skin_validator
        _validator_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "skin_validator.py")
        _v_mtime = os.path.getmtime(_validator_file) if os.path.exists(_validator_file) else 0
        if skin_validator is None or getattr(skin_validator, "_last_loaded_mtime", 0) < _v_mtime:
            try:
                import importlib
                import src.skin_validator
                importlib.reload(src.skin_validator)
                with open("config.yaml", "r") as _f:
                    _cfg = yaml.safe_load(_f)
                skin_validator = src.skin_validator.SkinValidator(_cfg)
                skin_validator._last_loaded_mtime = _v_mtime
                print(f"[API] 🔄 Hot-reloaded newest SkinValidator into memory (mtime={_v_mtime})")
            except Exception as _e:
                print(f"[API] ⚠ Skin validator reload error: {_e}")

        image_type = "unknown"
        if skin_validator is not None:
            validation = skin_validator.validate(temp_path)
            category = validation.get("category", "lesion")
            image_type = validation.get("image_type", "normal_camera")
            print(f"[API PREDICT GATE] File={file.filename} -> Category={category.upper()} | Scores={validation.get('category_scores')}")

            # Tier 1: Non-Skin / Objects / Memes
            if category == "non_skin":
                quality_result = quality_assessor.assess(temp_path)
                os.remove(temp_path)
                image_quality = ImageQuality(**quality_result.to_dict()) if quality_result else None
                return PredictionResponse(
                    label="None",
                    confidence=1.0,
                    all_probabilities={"Acne": 0.0, "Eczema": 0.0, "Herpes": 0.0},
                    device="cpu",
                    architecture="clip",
                    image_type="non_skin",
                    image_quality=image_quality,
                    is_inconclusive=False,
                    clinical_feedback="Non-skin or non-dermatological image detected. Please upload a clear photo of human skin.",
                )

            # Tier 2: Full Body / Portrait / Distant Photos
            if category == "full_body_portrait":
                quality_result = quality_assessor.assess(temp_path)
                os.remove(temp_path)
                image_quality = ImageQuality(**quality_result.to_dict()) if quality_result else None
                return PredictionResponse(
                    label="None",
                    confidence=1.0,
                    all_probabilities={"Acne": 0.0, "Eczema": 0.0, "Herpes": 0.0},
                    device="cpu",
                    architecture="clip",
                    image_type="full_body_portrait",
                    image_quality=image_quality,
                    is_inconclusive=False,
                    clinical_feedback="Full body or distant portrait photo detected. Please upload a close-up, focused photo of the specific skin lesion area.",
                )

            # Tier 3: Clear / Healthy Skin without Lesions (or jewelry on healthy skin)
            if category == "healthy_skin":
                quality_result = quality_assessor.assess(temp_path)
                os.remove(temp_path)
                image_quality = ImageQuality(**quality_result.to_dict()) if quality_result else None
                return PredictionResponse(
                    label="Clear",
                    confidence=0.98,
                    all_probabilities={"Acne": 0.0, "Eczema": 0.0, "Herpes": 0.0},
                    device="cpu",
                    architecture="clip",
                    image_type=image_type,
                    image_quality=image_quality,
                    is_inconclusive=False,
                    clinical_feedback="No active skin lesions detected. The skin appears healthy and unblemished.",
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

        # ---- Inconclusive & Low Confidence Safety Guard ----
        raw_label = result["label"]
        raw_conf = result["confidence"]
        all_probs = result["all_probabilities"]

        # Sort probability values descending
        sorted_probs = sorted(all_probs.values(), reverse=True)
        top_margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else sorted_probs[0]

        # Guard triggers if top class confidence is < 58% or top-two difference is < 10% (flat/ambiguous distribution)
        is_inconclusive = (raw_conf < 0.58) or (top_margin < 0.10)

        if is_inconclusive:
            final_label = "Inconclusive"
            clinical_feedback = (
                "This skin scan could not be matched with high certainty to our 3 priority conditions "
                "(Acne, Eczema, Herpes). Please consult a licensed dermatologist for comprehensive evaluation."
            )
        else:
            final_label = raw_label
            clinical_feedback = None

        return PredictionResponse(
            label=final_label,
            confidence=raw_conf,
            all_probabilities=all_probs,
            device=str(predictor.device),
            architecture=predictor.architecture,
            image_type=image_type,
            image_quality=image_quality,
            is_inconclusive=is_inconclusive,
            clinical_feedback=clinical_feedback,
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


# ============================================================
# Model Retraining & Management Endpoints
# ============================================================

@app.post("/train/start", response_model=dict)
async def start_training(request: StartTrainingRequest):
    """Trigger asynchronous background model retraining."""
    success = training_manager.start_training(
        architecture=request.architecture,
        epochs=request.epochs or 5,
        sync_dataset=request.sync_dataset if request.sync_dataset is not None else True,
        learning_rate=request.learning_rate,
    )
    if not success:
        raise HTTPException(status_code=409, detail="A training session is already in progress.")

    return {
        "message": f"Retraining started for {request.architecture or 'ensemble'}",
        "status": "started",
        "epochs": request.epochs or 5,
    }


@app.get("/train/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get live training status, progress percentage, loss, and logs."""
    return TrainingStatusResponse(**training_manager.get_status())


@app.post("/train/cancel", response_model=dict)
async def cancel_training():
    """Cancel current training session safely."""
    success = training_manager.cancel_training()
    if not success:
        current_status = training_manager.get_status().get("status")
        if current_status in ["cancelling", "cancelled", "idle", "completed", "failed"]:
            return {"message": "Training is not actively running.", "status": current_status}
        raise HTTPException(status_code=400, detail="No active training session to cancel.")
    return {"message": "Cancellation request submitted.", "status": "cancelling"}


@app.get("/model/stats", response_model=dict)
async def get_model_stats():
    """Return dataset statistics, baseline count, and model configuration."""
    return training_manager.get_dataset_stats()


@app.post("/dataset/sync", response_model=dict)
async def sync_dataset():
    """Manually synchronize dataset images from webapp storage."""
    count = training_manager.sync_gathered_dataset()
    return {"message": f"Successfully synced {count} images from webapp storage.", "copied_count": count}


@app.post("/model/reload", response_model=dict)
async def trigger_reload():
    """Hot-reload model predictor weights."""
    success = reload_predictor()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reload model predictor.")
    return {"message": "Model predictor reloaded successfully.", "status": "ready"}


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
