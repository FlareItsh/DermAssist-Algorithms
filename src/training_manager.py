"""
DermAssist - Training Manager & Validation Guard
=================================================
Manages asynchronous background training, dataset synchronization,
real-time progress tracking, and accuracy safety checks (Validation Guard).
"""

import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, create_dataloaders
from src.model import build_model


class TrainingManager:
    """
    Thread-safe manager for background model training with real-time status
    and automatic Validation Guard to protect model accuracy.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._current_stop_event: Optional[threading.Event] = None
        self._current_session_id: int = 0

        # Callbacks for predictor reload
        self._on_model_promoted_callbacks: List[Any] = []
        self.last_start_error: Optional[str] = None

        # State storage
        self.state: Dict[str, Any] = {
            "status": "idle",  # idle, syncing, training, evaluating, completed, failed, cancelled
            "progress": 0.0,
            "architecture": "swin_transformer",
            "current_epoch": 0,
            "total_epochs": 0,
            "current_batch": 0,
            "total_batches": 0,
            "train_loss": 0.0,
            "train_acc": 0.0,
            "val_loss": 0.0,
            "val_acc": 0.0,
            "baseline_val_acc": 0.0,
            "best_val_acc": 0.0,
            "model_promoted": False,
            "message": "Ready to train",
            "eta_seconds": 0,
            "elapsed_seconds": 0,
            "logs": [],
            "history": {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
            },
            "started_at": None,
            "completed_at": None,
        }

    def register_reload_callback(self, callback: Any):
        """Register a callback function to be called when a new model is promoted to production."""
        self._on_model_promoted_callbacks.append(callback)

    @property
    def _stop_requested(self) -> threading.Event:
        """Backward-compatibility shim for external inspections."""
        return self._current_stop_event if self._current_stop_event is not None else threading.Event()

    def log(self, message: str):
        """Append a log message thread-safely."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        try:
            print(f"[TrainingManager] {entry}")
        except UnicodeEncodeError:
            safe_entry = entry.encode("ascii", errors="replace").decode("ascii")
            print(f"[TrainingManager] {safe_entry}")
        with self._lock:
            self.state["logs"].append(entry)
            if len(self.state["logs"]) > 200:
                self.state["logs"].pop(0)

    def get_status(self) -> Dict[str, Any]:
        """Return a copy of the current training status."""
        with self._lock:
            # Update elapsed time if training
            if self.state["status"] in ["syncing", "training", "evaluating"] and self.state.get("start_time"):
                self.state["elapsed_seconds"] = int(time.time() - self.state["start_time"])
            return dict(self.state)

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Inspect baseline dataset images and current model information."""
        config = load_config(self.config_path)
        raw_dir = config["data"]["raw_dir"]

        classes_info = {}
        total_images = 0
        if os.path.exists(raw_dir):
            for item in os.listdir(raw_dir):
                item_path = os.path.join(raw_dir, item)
                if os.path.isdir(item_path):
                    count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                    classes_info[item] = count
                    total_images += count

        # Check production models available
        prod_dir = "models/production"
        models_available = []
        if os.path.exists(prod_dir):
            for f in os.listdir(prod_dir):
                if f.endswith(".pth"):
                    models_available.append(f)

        ensemble_enabled = config.get("inference", {}).get("ensemble_enabled", False)
        active_arch = (
            "Ensemble (Swin + ResNet50 + EfficientNet-V2)"
            if ensemble_enabled
            else config.get("inference", {}).get("active_inference_model", "swin_transformer")
        )

        return {
            "total_baseline_images": total_images,
            "classes": classes_info,
            "active_architecture": active_arch,
            "ensemble_enabled": ensemble_enabled,
            "models_available": models_available,
        }

    def sync_gathered_dataset(
        self,
        laravel_dataset_path: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        """
        Synchronize newly collected scan images from Laravel's dataset storage
        into the algorithm's data/raw/ directories.
        """
        self.log("Syncing newly gathered dataset images from webapp storage...")
        if not laravel_dataset_path:
            # Default to standard project sibling directory
            laravel_dataset_path = os.path.abspath(
                os.path.join(PROJECT_ROOT, "..", "DermAssist-API", "storage", "app", "public", "dataset")
            )

        if not os.path.exists(laravel_dataset_path):
            self.log(f"⚠ Dataset path not found: {laravel_dataset_path}. Skipping sync.")
            return 0

        config = load_config(self.config_path)
        target_raw_dir = config["data"]["raw_dir"]
        os.makedirs(target_raw_dir, exist_ok=True)

        # Mapping lowercase category names to TitleCase in data/raw
        category_map = {
            "acne": "Acne",
            "eczema": "Eczema",
            "herpes": "Herpes",
        }

        copied_count = 0
        deleted_count = 0
        event = stop_event or self._current_stop_event

        for cat_lower, target_class_name in category_map.items():
            # Respect cancellation during potentially slow file sync
            if event and event.is_set():
                self.log("⚠ Dataset sync aborted by cancellation request.")
                return copied_count

            source_folder = os.path.join(laravel_dataset_path, cat_lower)
            target_folder = os.path.join(target_raw_dir, target_class_name)
            os.makedirs(target_folder, exist_ok=True)

            valid_source_fnames = set()
            if os.path.exists(source_folder) and os.path.isdir(source_folder):
                valid_source_fnames = set(os.listdir(source_folder))
                for fname in valid_source_fnames:
                    if event and event.is_set():
                        self.log("⚠ Dataset sync aborted by cancellation request.")
                        return copied_count

                    src_file = os.path.join(source_folder, fname)
                    if os.path.isfile(src_file):
                        dest_file = os.path.join(target_folder, f"webapp_{fname}")
                        if not os.path.exists(dest_file):
                            try:
                                shutil.copy2(src_file, dest_file)
                                copied_count += 1
                            except Exception as e:
                                self.log(f"Failed to copy {fname}: {e}")

            # Prune any previously synced webapp images that were deleted from webapp storage
            if os.path.exists(target_folder):
                for target_file in os.listdir(target_folder):
                    if event and event.is_set():
                        self.log("⚠ Dataset sync aborted by cancellation request.")
                        return copied_count
                    if target_file.startswith("webapp_"):
                        orig_fname = target_file[7:]  # strip 'webapp_'
                        if orig_fname not in valid_source_fnames:
                            full_target_path = os.path.join(target_folder, target_file)
                            try:
                                os.remove(full_target_path)
                                deleted_count += 1
                                self.log(f"Purged deleted image from training set: {target_file}")
                            except Exception as e:
                                self.log(f"Failed to remove orphaned image {target_file}: {e}")

        self.log(
            f"Dataset sync completed: {copied_count} new images integrated, "
            f"{deleted_count} deleted images pruned from training dataset."
        )
        return copied_count

    def sync_expansion_disease(
        self,
        disease_slug: str,
        laravel_oos_path: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        """
        Synchronize images of a selected out-of-scope disease into data/raw/ for model expansion.
        """
        self.log(f"Syncing out-of-scope disease '{disease_slug}' into training dataset...")
        if not laravel_oos_path:
            laravel_oos_path = os.path.abspath(
                os.path.join(PROJECT_ROOT, "..", "DermAssist-API", "storage", "app", "public", "out_of_scope_dataset")
            )

        if not os.path.exists(laravel_oos_path):
            self.log(f"⚠ Out-of-scope dataset path not found: {laravel_oos_path}")
            return 0

        config = load_config(self.config_path)
        target_raw_dir = config["data"]["raw_dir"]
        os.makedirs(target_raw_dir, exist_ok=True)

        source_folder = os.path.join(laravel_oos_path, disease_slug.lower())
        target_folder = os.path.join(target_raw_dir, disease_slug.replace("_", " ").title())
        os.makedirs(target_folder, exist_ok=True)

        copied_count = 0
        event = stop_event or self._current_stop_event
        if os.path.exists(source_folder) and os.path.isdir(source_folder):
            for fname in os.listdir(source_folder):
                if event and event.is_set():
                    return copied_count
                src_file = os.path.join(source_folder, fname)
                if os.path.isfile(src_file):
                    dest_file = os.path.join(target_folder, f"webapp_{fname}")
                    if not os.path.exists(dest_file):
                        try:
                            shutil.copy2(src_file, dest_file)
                            copied_count += 1
                        except Exception as e:
                            self.log(f"Failed to copy {fname}: {e}")

        self.log(f"Synced {copied_count} images for new disease '{disease_slug.title()}'.")
        return copied_count

    def _set_state(
        self,
        stop_event: Optional[threading.Event] = None,
        session_id: Optional[int] = None,
        **kwargs
    ):
        """Thread-safely update state unless cancellation was requested or session expired."""
        with self._lock:
            event = stop_event or self._current_stop_event
            if event and event.is_set():
                return
            if session_id is not None and session_id != self._current_session_id:
                return
            self.state.update(kwargs)

    def start_training(
        self,
        architecture: Optional[str] = None,
        epochs: int = 5,
        sync_dataset: bool = True,
        learning_rate: Optional[float] = None,
        expansion_disease: Optional[str] = None,
    ) -> bool:
        """
        Start the training loop in a non-blocking background thread.
        Guarantees strictly ONE active training thread at any time.
        """
        self.last_start_error = None

        # Step 0: Validate dataset sufficiency for model expansion runs
        if expansion_disease:
            disease_slug = expansion_disease.lower().strip()
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

            # Count in Laravel public storage
            laravel_oos_path = os.path.abspath(
                os.path.join(PROJECT_ROOT, "..", "DermAssist-API", "storage", "app", "public", "out_of_scope_dataset", disease_slug)
            )
            oos_count = 0
            if os.path.exists(laravel_oos_path) and os.path.isdir(laravel_oos_path):
                oos_count = len([f for f in os.listdir(laravel_oos_path) if os.path.splitext(f)[1].lower() in valid_exts])

            # Count in existing raw_dir
            config = load_config(self.config_path)
            raw_dir = config["data"]["raw_dir"]
            raw_disease_folder = os.path.join(raw_dir, expansion_disease.replace("_", " ").title())
            raw_count = 0
            if os.path.exists(raw_disease_folder) and os.path.isdir(raw_disease_folder):
                raw_count = len([
                    f for f in os.listdir(raw_disease_folder)
                    if os.path.splitext(f)[1].lower() in valid_exts and not f.lower().startswith("aug_")
                ])

            available_count = max(oos_count, raw_count)
            if available_count < 10:
                err_msg = (
                    f"Insufficient dataset for model expansion: At least 10 verified research images "
                    f"are required for {expansion_disease.title()}. Currently {available_count} available."
                )
                self.last_start_error = err_msg
                self.log(f"⚠ {err_msg}")
                return False

        # Step 1: Handle any existing running thread
        if self._thread is not None and self._thread.is_alive():
            # If the active thread was requested to stop, wait for it to cleanly terminate
            if self._current_stop_event is not None and self._current_stop_event.is_set():
                self.log("⏳ Waiting for previous training session to complete shutdown...")
                self._thread.join(timeout=4.0)

            # If still alive, strictly refuse to start a second concurrent training run
            if self._thread.is_alive():
                self.log("⚠ Cannot start new training: previous training session is still active.")
                return False

        with self._lock:
            self._current_session_id += 1
            session_id = self._current_session_id
            session_stop_event = threading.Event()
            self._current_stop_event = session_stop_event

            arch = architecture or "swin_transformer"
            self.state.update({
                "status": "syncing" if sync_dataset else "training",
                "progress": 0.0,
                "architecture": arch,
                "current_epoch": 0,
                "total_epochs": epochs,
                "current_batch": 0,
                "total_batches": 0,
                "train_loss": 0.0,
                "train_acc": 0.0,
                "val_loss": 0.0,
                "val_acc": 0.0,
                "baseline_val_acc": 0.0,
                "best_val_acc": 0.0,
                "model_promoted": False,
                "message": (
                    f"Initializing expansion training for {expansion_disease.title()}..."
                    if expansion_disease
                    else "Initializing training pipeline..."
                ),
                "eta_seconds": 0,
                "elapsed_seconds": 0,
                "logs": [],
                "history": {
                    "train_loss": [],
                    "train_acc": [],
                    "val_loss": [],
                    "val_acc": [],
                },
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "start_time": time.time(),
                "expansion_disease": expansion_disease,
            })

        self._thread = threading.Thread(
            target=self._run_training_worker,
            args=(arch, epochs, sync_dataset, learning_rate, expansion_disease, session_stop_event, session_id),
            daemon=True,
        )
        self._thread.start()
        return True

    def cancel_training(self) -> bool:
        """
        Request training cancellation immediately and cleanly signal background workers.
        Transitions status: active -> cancelling -> cancelled.
        """
        active_thread = None
        with self._lock:
            if self._current_stop_event is None and (self._thread is None or not self._thread.is_alive()):
                return False

            if self._current_stop_event is not None:
                self._current_stop_event.set()

            active_thread = self._thread
            self.state["status"] = "cancelling"
            self.state["message"] = "Stopping training pipeline..."

        self.log("🛑 Training cancellation requested. Signaling background worker to abort...")

        # Fast join: if thread terminates quickly (between steps/batches), immediately mark cancelled
        if active_thread is not None and active_thread.is_alive():
            active_thread.join(timeout=1.5)

        with self._lock:
            if active_thread is None or not active_thread.is_alive():
                self.state["status"] = "cancelled"
                self.state["message"] = "Training was cancelled."
                self.state["completed_at"] = datetime.now().isoformat()
                self.log("🛑 Training successfully cancelled.")

        return True

    def _evaluate_baseline(
        self,
        model: nn.Module,
        val_loader,
        criterion,
        device,
        stop_event: Optional[threading.Event] = None,
    ) -> tuple:
        """Evaluate baseline accuracy before training begins."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                if stop_event and stop_event.is_set():
                    return 0.0, 0.0
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy

    def _run_training_worker(
        self,
        architecture: str,
        epochs: int,
        sync_dataset: bool,
        learning_rate: Optional[float],
        expansion_disease: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
        session_id: Optional[int] = None,
    ):
        """Worker thread executing the training loop with Validation Guard."""
        current_stop = stop_event or self._current_stop_event or threading.Event()
        try:
            if current_stop.is_set():
                self._handle_cancellation(session_id)
                return

            # Determine if retraining full ensemble or a single backbone
            if architecture in ["ensemble", "all", "tri_model"]:
                arch_list = ["swin_transformer", "resnet50", "efficientnet_v2"]
                is_ensemble_run = True
            else:
                arch_list = [architecture]
                is_ensemble_run = False

            total_models = len(arch_list)
            mode_desc = f"Expansion Run (+{expansion_disease.title()})" if expansion_disease else "Standard 3-Disease Retraining"
            self.log(
                f"Starting {'Ensemble' if is_ensemble_run else 'Single'} {mode_desc}: "
                f"Models={', '.join([a.upper() for a in arch_list])}, Epochs/Model={epochs}"
            )

            # Configure target classes explicitly to guarantee zero conflict between standard retraining and expansion
            if expansion_disease:
                disease_title = expansion_disease.replace("_", " ").title()
                raw_dir = load_config(self.config_path)["data"]["raw_dir"]
                disease_dir = os.path.join(raw_dir, disease_title)
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
                curr_images = (
                    len([f for f in os.listdir(disease_dir) if os.path.splitext(f)[1].lower() in valid_exts and not f.startswith("aug_")])
                    if os.path.exists(disease_dir)
                    else 0
                )
                if curr_images < 10:
                    abort_err = f"Aborted: Insufficient dataset for {disease_title} ({curr_images}/10 images). At least 10 verified images are required."
                    self.log(f"❌ {abort_err}")
                    with self._lock:
                        if session_id is None or session_id == self._current_session_id:
                            self.state["status"] = "failed"
                            self.state["message"] = abort_err
                            self.state["completed_at"] = datetime.now().isoformat()
                    return

                target_class_names = ["Acne", "Eczema", "Herpes", disease_title]
                target_num_classes = 4
            else:
                target_class_names = ["Acne", "Eczema", "Herpes"]
                target_num_classes = 3
                # Guarantee isolation: prune non-baseline folders from data/raw for pure 3-disease retraining
                raw_dir = load_config(self.config_path)["data"]["raw_dir"]
                if os.path.exists(raw_dir):
                    for folder_name in os.listdir(raw_dir):
                        if folder_name not in target_class_names and os.path.isdir(os.path.join(raw_dir, folder_name)):
                            shutil.rmtree(os.path.join(raw_dir, folder_name), ignore_errors=True)
                            self.log(f"🧹 Cleaned non-baseline folder '{folder_name}' to maintain pure 3-disease training.")

            if current_stop.is_set():
                self._handle_cancellation(session_id)
                return

            # Step 1: Sync dataset if requested
            if sync_dataset:
                self.sync_gathered_dataset(stop_event=current_stop)
                if expansion_disease:
                    self.sync_expansion_disease(expansion_disease, stop_event=current_stop)

            # 🛡️ Dual-Safekeeping: When expanding, archive 3-disease baseline models
            if expansion_disease:
                backup_3class_dir = "models/production_3class_backup"
                os.makedirs(backup_3class_dir, exist_ok=True)
                prod_dir = "models/production"
                if os.path.exists(prod_dir):
                    for f in os.listdir(prod_dir):
                        if f.endswith(".pth") and not f.endswith("_backup.pth"):
                            shutil.copy2(os.path.join(prod_dir, f), os.path.join(backup_3class_dir, f))
                    self.log(f"🛡️ Safekeeping: 3-Disease baseline models safely archived in {backup_3class_dir}/")

            if current_stop.is_set():
                self._handle_cancellation(session_id)
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"Compute device: {device}")

            promoted_models = []
            results_summary = []

            for arch_idx, arch in enumerate(arch_list):
                if current_stop.is_set():
                    self._handle_cancellation(session_id)
                    return

                model_num = arch_idx + 1
                self.log("=" * 60)
                self.log(f"▶ [{model_num}/{total_models}] TRAINING BACKBONE: {arch.upper()}")
                self.log("=" * 60)

                self._set_state(
                    stop_event=current_stop,
                    session_id=session_id,
                    status="training",
                    architecture=arch,
                    message=(
                        f"Training Model {model_num}/{total_models} ({arch.replace('_', ' ').title()})..."
                        if is_ensemble_run
                        else f"Training {arch.replace('_', ' ').title()}..."
                    )
                )

                config = load_config(self.config_path)
                config["advanced"]["active_architecture"] = arch
                config["training"]["epochs"] = epochs
                config["model"]["num_classes"] = target_num_classes
                config["model"]["class_names"] = target_class_names
                if learning_rate is not None:
                    config["training"]["learning_rate"] = learning_rate

                # Step 2: Dataloaders
                self.log(f"Building clinical data loaders for {arch.upper()} with classes: {target_class_names}...")
                train_loader, val_loader, class_names = create_dataloaders(config)
                total_batches = len(train_loader)

                self._set_state(
                    stop_event=current_stop,
                    session_id=session_id,
                    total_batches=total_batches
                )

                if current_stop.is_set():
                    self._handle_cancellation(session_id)
                    return

                # Step 3: Model Setup
                self.log(f"Initializing {arch.upper()} architecture...")
                model = build_model(config, device)

                prod_model_path = os.path.join("models/production", f"best_model_{arch}.pth")
                if not os.path.exists(prod_model_path):
                    prod_model_path = os.path.join("models/production", "best_model.pth")

                baseline_acc = 0.0
                if os.path.exists(prod_model_path):
                    try:
                        self.log(f"Loading existing weights from: {prod_model_path}")
                        ckpt = torch.load(prod_model_path, map_location=device)
                        state_dict = ckpt.get("model_state_dict", ckpt)
                        model.load_state_dict(state_dict, strict=False)
                        baseline_acc = ckpt.get("val_acc", 0.0)
                        self.log(f"Prior checkpoint validation accuracy: {baseline_acc:.2f}%")
                    except Exception as e:
                        self.log(f"Could not load checkpoint ({e}). Training from backbone initialization.")

                if current_stop.is_set():
                    self._handle_cancellation(session_id)
                    return

                # Calculate class weights
                from collections import Counter
                train_labels = [
                    train_loader.dataset.subset.dataset.samples[i][1]
                    for i in train_loader.dataset.subset.indices
                ]
                class_counts = Counter(train_labels)
                total_train = len(train_labels)
                num_classes = len(class_names)
                weights = [total_train / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
                class_weights = torch.FloatTensor(weights).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)

                # Baseline evaluation on current validation set
                if baseline_acc == 0.0 and len(val_loader) > 0:
                    self.log(f"Evaluating baseline for {arch.upper()} on validation set...")
                    self._set_state(
                        stop_event=current_stop,
                        session_id=session_id,
                        status="evaluating",
                        message=f"Measuring baseline for {arch.replace('_', ' ').title()}..."
                    )
                    _, baseline_acc = self._evaluate_baseline(model, val_loader, criterion, device, stop_event=current_stop)
                    if current_stop.is_set():
                        self._handle_cancellation(session_id)
                        return
                    self.log(f"Measured baseline accuracy: {baseline_acc:.2f}%")

                self._set_state(
                    stop_event=current_stop,
                    session_id=session_id,
                    baseline_val_acc=round(baseline_acc, 2),
                    status="training"
                )

                # Setup Optimizer & Scheduler
                lr = learning_rate or (0.00005 if arch == "swin_transformer" else 0.0001)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

                session_best_val_acc = 0.0
                best_val_acc = baseline_acc
                new_best_model_weights = None
                total_steps = epochs * total_batches
                step_count = 0

                # Training Epochs
                for epoch in range(epochs):
                    if current_stop.is_set():
                        self._handle_cancellation(session_id)
                        return

                    epoch_start_time = time.time()
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    self._set_state(
                        stop_event=current_stop,
                        session_id=session_id,
                        current_epoch=epoch + 1,
                        message=(
                            f"[{model_num}/{total_models}] {arch.replace('_', ' ').title()} - Epoch {epoch + 1}/{epochs}"
                            if is_ensemble_run
                            else f"Training Epoch {epoch + 1}/{epochs}"
                        )
                    )

                    for batch_idx, (images, labels) in enumerate(train_loader):
                        if current_stop.is_set():
                            self._handle_cancellation(session_id)
                            return

                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        if current_stop.is_set():
                            self._handle_cancellation(session_id)
                            return

                        running_loss += loss.item() * images.size(0)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        step_count += 1

                        if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                            if current_stop.is_set():
                                self._handle_cancellation(session_id)
                                return

                            # Calculate cumulative progress across all models in ensemble
                            sub_progress = (step_count / max(total_steps, 1)) * 100.0
                            overall_progress = round(
                                ((arch_idx * 100.0) + sub_progress) / total_models, 1
                            )
                            current_train_acc = 100.0 * correct / max(total, 1)
                            current_train_loss = running_loss / max(total, 1)

                            elapsed = time.time() - self.state["start_time"]
                            total_estimated_steps = total_steps * total_models
                            cumulative_steps = (arch_idx * total_steps) + step_count
                            rate = cumulative_steps / max(elapsed, 0.1)
                            remaining_steps = total_estimated_steps - cumulative_steps
                            eta = int(remaining_steps / max(rate, 0.001))

                            self._set_state(
                                stop_event=current_stop,
                                session_id=session_id,
                                progress=min(overall_progress, 99.0),
                                current_batch=batch_idx + 1,
                                train_loss=round(current_train_loss, 4),
                                train_acc=round(current_train_acc, 2),
                                eta_seconds=eta
                            )

                    # Validation Phase
                    if current_stop.is_set():
                        self._handle_cancellation(session_id)
                        return

                    scheduler.step()
                    model.eval()
                    val_loss_sum = 0.0
                    val_correct = 0
                    val_total = 0

                    self._set_state(
                        stop_event=current_stop,
                        session_id=session_id,
                        status="evaluating",
                        message=f"Validating {arch.replace('_', ' ').title()} (Epoch {epoch + 1}/{epochs})..."
                    )

                    with torch.no_grad():
                        for images, labels in val_loader:
                            if current_stop.is_set():
                                break

                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_loss_sum += loss.item() * images.size(0)
                            _, predicted = outputs.max(1)
                            val_total += labels.size(0)
                            val_correct += predicted.eq(labels).sum().item()

                    if current_stop.is_set():
                        self._handle_cancellation(session_id)
                        return

                    epoch_train_loss = running_loss / max(total, 1)
                    epoch_train_acc = 100.0 * correct / max(total, 1)
                    epoch_val_loss = val_loss_sum / max(val_total, 1)
                    epoch_val_acc = 100.0 * val_correct / max(val_total, 1)

                    if epoch_val_acc > session_best_val_acc:
                        session_best_val_acc = epoch_val_acc

                    is_beat_baseline = epoch_val_acc >= baseline_acc
                    is_new_best = epoch_val_acc > best_val_acc
                    if is_beat_baseline and is_new_best:
                        best_val_acc = epoch_val_acc
                        new_best_model_weights = {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": epoch_val_acc,
                            "class_names": class_names,
                            "architecture": arch,
                            "timestamp": datetime.now().isoformat(),
                        }

                    epoch_duration = int(time.time() - epoch_start_time)
                    self.log(
                        f"[{arch.upper()}] Epoch {epoch + 1}/{epochs} ({epoch_duration}s) | "
                        f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
                        f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%"
                        + (" ⭐ (Beat Baseline)" if is_beat_baseline else "")
                    )

                    with self._lock:
                        if not current_stop.is_set() and (session_id is None or session_id == self._current_session_id):
                            self.state["status"] = "training"
                            self.state["val_loss"] = round(epoch_val_loss, 4)
                            self.state["val_acc"] = round(epoch_val_acc, 2)
                            self.state["best_val_acc"] = round(max(session_best_val_acc, baseline_acc), 2)
                            self.state["history"]["train_loss"].append(round(epoch_train_loss, 4))
                            self.state["history"]["train_acc"].append(round(epoch_train_acc, 2))
                            self.state["history"]["val_loss"].append(round(epoch_val_loss, 4))
                            self.state["history"]["val_acc"].append(round(epoch_val_acc, 2))

                # Step 4: VALIDATION GUARD CHECK FOR THIS MODEL
                if current_stop.is_set():
                    self._handle_cancellation(session_id)
                    return

                self.log(f"--- VALIDATION GUARD: {arch.upper()} ---")
                self.log(f"Baseline: {baseline_acc:.2f}% | Session Best: {session_best_val_acc:.2f}%")

                target_prod_path = os.path.join("models/production", f"best_model_{arch}.pth")
                backup_prod_path = os.path.join("models/production", f"best_model_{arch}_backup.pth")
                os.makedirs("models/production", exist_ok=True)

                if new_best_model_weights is not None and session_best_val_acc >= baseline_acc:
                    self.log(f"✅ PASSED GUARD: {arch.upper()} improved ({baseline_acc:.2f}% -> {session_best_val_acc:.2f}%).")
                    if os.path.exists(target_prod_path):
                        shutil.copy2(target_prod_path, backup_prod_path)
                    torch.save(new_best_model_weights, target_prod_path)
                    if expansion_disease:
                        checkpoints_4class_dir = "models/checkpoints_4class"
                        os.makedirs(checkpoints_4class_dir, exist_ok=True)
                        ckpt_4class_path = os.path.join(checkpoints_4class_dir, f"best_model_{arch}_4class.pth")
                        torch.save(new_best_model_weights, ckpt_4class_path)
                        self.log(f"💾 Safe-keeping Checkpoint: Expanded model saved -> {ckpt_4class_path}")
                    self.log(f"🚀 Deployed to production -> {target_prod_path}")
                    promoted_models.append(arch)
                    results_summary.append(f"{arch.upper()}: {session_best_val_acc:.2f}% (Promoted)")
                else:
                    self.log(f"🛡 GUARD PRESERVED: {arch.upper()} session best ({session_best_val_acc:.2f}%) did not beat baseline ({baseline_acc:.2f}%).")
                    results_summary.append(f"{arch.upper()}: Baseline {baseline_acc:.2f}% kept (Session: {session_best_val_acc:.2f}%)")

            if current_stop.is_set():
                self._handle_cancellation(session_id)
                return

            # Final Step: Hot-reload predictor if any models promoted
            if promoted_models:
                self.log("=" * 60)
                self.log(f"Hot-reloading live ensemble with updated weights for: {', '.join(promoted_models)}")
                for cb in self._on_model_promoted_callbacks:
                    try:
                        cb()
                    except Exception as ex:
                        self.log(f"⚠ Predictor reload callback failed: {ex}")

            with self._lock:
                if not current_stop.is_set() and (session_id is None or session_id == self._current_session_id):
                    self.state["status"] = "completed"
                    self.state["progress"] = 100.0
                    self.state["model_promoted"] = len(promoted_models) > 0
                    self.state["completed_at"] = datetime.now().isoformat()
                    self.state["architecture"] = "ensemble" if is_ensemble_run else arch_list[0]
                    summary_str = " | ".join(results_summary)
                    self.state["message"] = (
                        f"Ensemble training complete! {len(promoted_models)}/{total_models} models improved. [{summary_str}]"
                        if is_ensemble_run
                        else f"Training complete! {results_summary[0]}"
                    )

        except Exception as e:
            if current_stop.is_set():
                self._handle_cancellation(session_id)
                return

            import traceback
            err_msg = traceback.format_exc()
            self.log(f"❌ Training failed with exception: {e}")
            print(err_msg)
            with self._lock:
                if session_id is None or session_id == self._current_session_id:
                    self.state["status"] = "failed"
                    self.state["message"] = f"Training failed: {str(e)}"
                    self.state["completed_at"] = datetime.now().isoformat()

    def _handle_cancellation(self, session_id: Optional[int] = None):
        """Clean up when cancellation is requested."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        with self._lock:
            if session_id is None or session_id == self._current_session_id:
                self.state["status"] = "cancelled"
                self.state["message"] = "Training was cancelled."
                self.state["completed_at"] = datetime.now().isoformat()
        self.log("🛑 Training safely aborted.")


# Global Singleton
training_manager = TrainingManager()
