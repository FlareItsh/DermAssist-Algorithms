"""
DermAssist AI — Modal Serverless Deployment
============================================
Hosts the PyTorch FastAPI inference & training engine on Modal serverless GPUs/CPUs.

Deployment:
    modal deploy modal_app.py

Local testing:
    modal serve modal_app.py
"""

import os
import sys
import modal

APP_NAME = "dermassist-ai"

# Define Modal App
app = modal.App(APP_NAME)

# Define container image with all deep learning dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "python-multipart>=0.0.6",
        "Pillow>=10.0.0",
        "opencv-python-headless>=4.8.0",
        "transformers>=4.35.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    )
    .run_commands(
        # Pre-download CLIP model during image build to avoid cold-start delays
        "python3 -c 'from transformers import CLIPProcessor, CLIPModel; CLIPProcessor.from_pretrained(\"openai/clip-vit-base-patch32\"); CLIPModel.from_pretrained(\"openai/clip-vit-base-patch32\")'"
    )
    .add_local_dir(
        os.path.dirname(os.path.abspath(__file__)),
        remote_path="/root/algorithms",
        ignore=[
            "venv",
            "__pycache__",
            ".git",
            "scratch",
            "*.log",
            "data",
            "models/checkpoints",
            "models/checkpoints_*",
            "*.png",
        ],
    )
)

# Volume for persisting retrained checkpoints across runs
volume = modal.Volume.from_name("dermassist-models", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",  # NVIDIA T4 (16GB) — Cheapest GPU on Modal ($0.59/hr, billed per-second)
    timeout=600,  # 10 min timeout for retraining jobs
    scaledown_window=300,  # Automatically scale down to 0 after 5 min idle ($0 idle cost)
    volumes={"/root/algorithms/models/checkpoints": volume},
)
@modal.asgi_app()
def fastapi_app():
    """Wrap FastAPI application for Modal ASGI hosting."""
    base_dir = "/root/algorithms"
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    os.chdir(base_dir)

    from api.app import app as web_app

    return web_app
