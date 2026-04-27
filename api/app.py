"""
!!! SYSTEM FLARE: API/APP.PY IS STARTING NOW !!!
================================================
"""
import os
import sys
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
from src.inference import SkinLesionPredictor

# ============================================================
# Schemas
# ============================================================

class Prediction(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    all_probabilities: dict
    device: str
    architecture: str

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
        
        # FORCE LEGACY LOCK for stability during training
        model_path = os.path.join("models/production", "best_3class_legacy.pth")
        active_arch = "resnet50"
        
        if not os.path.exists(model_path):
            print(f"[API] ❌ ERROR: Legacy model {model_path} missing!")
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
            
        # Inference
        result = predictor.predict(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            device=str(predictor.device),
            architecture=predictor.architecture
        )
    except Exception as e:
        import traceback
        print("\n" + "!"*60)
        print(" ❌ PREDICTION CRASH:")
        print(traceback.format_exc())
        print("!"*60 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
