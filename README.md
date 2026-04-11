# 🩺 DermAssist — Skin Lesion Detection

AI-powered skin lesion classification using **PyTorch** and **ResNet50** transfer learning. Designed to classify dermoscopic images into 7 skin disease categories.

---

## 📁 Project Structure

```
DermAssist-Algorithms/
├── data/
│   ├── raw/                    # Raw dataset (class subdirectories)
│   └── processed/              # Preprocessed data (auto-generated)
├── models/
│   ├── checkpoints/            # Training checkpoints per epoch
│   └── production/             # Best model for deployment
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Image preprocessing & augmentation
│   ├── model.py                # ResNet50 transfer learning architecture
│   ├── train.py                # Training loop with validation
│   └── inference.py            # Prediction logic for single images
├── api/
│   ├── __init__.py
│   └── app.py                  # FastAPI REST API wrapper
├── config.yaml                 # Hyperparameters & settings
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your images into class subdirectories inside `data/raw/`:

```
data/raw/
├── Actinic keratoses/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── Basal cell carcinoma/
│   └── ...
├── Benign keratosis/
│   └── ...
├── Dermatofibroma/
│   └── ...
├── Melanoma/
│   └── ...
├── Melanocytic nevi/
│   └── ...
└── Vascular lesions/
    └── ...
```

> **Tip:** The [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) is a great starting point with 10,015 dermoscopic images across these 7 classes.

### 3. Configure Hyperparameters

Edit `config.yaml` to adjust settings:

```yaml
training:
  batch_size: 32
  epochs: 25
  learning_rate: 0.001

model:
  num_classes: 7
  dropout_rate: 0.5
```

---

## 🏋️ Training

### Start Training

```bash
python -m src.train
```

Or with a custom config:

```bash
python -m src.train --config config.yaml
```

### What Happens During Training

1. **Data Loading** — Images are resized to 224×224, normalized with ImageNet stats, and augmented (random flips, rotation, color jitter).
2. **Model** — A pre-trained ResNet50 backbone with a custom classification head (Dropout → Linear → ReLU → BatchNorm → Dropout → Linear).
3. **Optimization** — Adam optimizer with StepLR scheduling and early stopping.
4. **Checkpointing** — The best model (by validation accuracy) is saved to `models/production/best_model.pth`.
5. **Visualization** — Training curves are saved to `training_history.png`.

### Training Output

```
══════════════════════════════════════════════════════════════
  DermAssist — Skin Lesion Detection Training
══════════════════════════════════════════════════════════════
  Device: cuda
  Epochs: 25
  Batch:  32
  LR:     0.001
══════════════════════════════════════════════════════════════

  Epoch   1/25 │ Train Loss: 1.2345  Acc:  45.23% │ Val Loss: 0.9876  Acc:  55.67%
  ★ Best model saved → models/production/best_model.pth (val_acc=55.67%)
  ...
```

---

## 🔍 Inference (Single Image)

### Command Line

```bash
python -m src.inference --image path/to/skin_image.jpg
```

### Options

```bash
python -m src.inference \
    --image path/to/image.jpg \
    --model models/production/best_model.pth \
    --config config.yaml \
    --top-k 5
```

### Python API

```python
from src.inference import load_predictor
from PIL import Image

# Load the predictor
predictor = load_predictor(config_path="config.yaml")

# Run prediction
image = Image.open("path/to/skin_image.jpg")
result = predictor.predict(image)

print(result)
# {
#     "label": "Melanoma",
#     "confidence": 0.9234,
#     "class_index": 4,
#     "all_probabilities": { ... }
# }
```

---

## 🌐 REST API

### Start the API Server

```bash
# Option 1: Using uvicorn directly
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Using the Python script
python api/app.py
```

The API will start at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint         | Description                        |
| ------ | ---------------- | ---------------------------------- |
| POST   | `/predict`       | Classify a skin lesion image       |
| POST   | `/predict/top-k` | Get top-K predictions              |
| GET    | `/health`        | Check API health and model status  |
| GET    | `/classes`       | List supported skin lesion classes |

### Example: POST /predict

Using **cURL**:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@path/to/skin_image.jpg"
```

**Response:**

```json
{
  "label": "Melanoma",
  "confidence": 0.9234,
  "class_index": 4,
  "all_probabilities": {
    "Actinic keratoses": 0.0021,
    "Basal cell carcinoma": 0.0103,
    "Benign keratosis": 0.0187,
    "Dermatofibroma": 0.0045,
    "Melanoma": 0.9234,
    "Melanocytic nevi": 0.0298,
    "Vascular lesions": 0.0112
  }
}
```

Using **Python requests**:

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("path/to/skin_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

---

## ⚙️ Configuration Reference

All settings are in `config.yaml`:

| Section  | Key                       | Default | Description                 |
| -------- | ------------------------- | ------- | --------------------------- |
| data     | `image_size`              | 224     | Input image resolution      |
| data     | `train_split`             | 0.8     | Train/val split ratio       |
| data     | `num_workers`             | 4       | DataLoader worker processes |
| model    | `num_classes`             | 7       | Number of disease classes   |
| model    | `dropout_rate`            | 0.5     | Dropout probability         |
| training | `batch_size`              | 32      | Training batch size         |
| training | `epochs`                  | 25      | Maximum training epochs     |
| training | `learning_rate`           | 0.001   | Initial learning rate       |
| training | `early_stopping_patience` | 5       | Epochs before early stop    |
| api      | `host`                    | 0.0.0.0 | API server host             |
| api      | `port`                    | 8000    | API server port             |

---

## 🛠️ Tech Stack

- **PyTorch** — Deep learning framework
- **torchvision** — Pre-trained ResNet50 + image transforms
- **FastAPI** — High-performance async REST API
- **Pillow** — Image loading and manipulation
- **scikit-learn** — Metrics and evaluation

---

## 📝 License

This project is part of the DermAssist platform.

---

_Built with ❤️ by the DermAssist team (FlareItsh)_
