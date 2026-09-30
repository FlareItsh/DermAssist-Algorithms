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
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()

        # Callbacks for predictor reload
        self._on_model_promoted_callbacks: List[Any] = []

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

    def log(self, message: str, replace_last: bool = False):
        """Append or update a log message thread-safely."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        print(f"[TrainingManager] {entry}")
        with self._lock:
            if replace_last and len(self.state["logs"]) > 0:
                self.state["logs"][-1] = entry
            else:
                self.state["logs"].append(entry)
                if len(self.state["logs"]) > 500:
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

    def sync_gathered_dataset(self, laravel_dataset_path: Optional[str] = None) -> int:
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

        for cat_lower, target_class_name in category_map.items():
            # Bug fix: respect cancellation during potentially slow file sync
            if self._stop_requested.is_set():
                self.log("⚠ Dataset sync aborted by cancellation request.")
                return copied_count

            source_folder = os.path.join(laravel_dataset_path, cat_lower)
            target_folder = os.path.join(target_raw_dir, target_class_name)
            os.makedirs(target_folder, exist_ok=True)

            valid_source_fnames = set()
            if os.path.exists(source_folder) and os.path.isdir(source_folder):
                valid_source_fnames = set(os.listdir(source_folder))
                for fname in valid_source_fnames:
                    if self._stop_requested.is_set():
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

    def start_training(
        self,
        architecture: Optional[str] = None,
        epochs: int = 5,
        sync_dataset: bool = True,
        learning_rate: Optional[float] = None,
    ) -> bool:
        """
        Start the training loop in a non-blocking background thread.
        """
        with self._lock:
            if self.state["status"] in ["syncing", "training", "evaluating"]:
                return False

            self._stop_requested.clear()
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
                "message": "Initializing training pipeline...",
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
            })

        self._thread = threading.Thread(
            target=self._run_training_worker,
            args=(arch, epochs, sync_dataset, learning_rate),
            daemon=True,
        )
        self._thread.start()
        return True

    def cancel_training(self, wait_seconds: float = 30.0) -> bool:
        """
        Request training cancellation safely and idempotently, and wait for the worker thread to terminate.

        The method signals the stop event, then joins the background thread with a
        generous timeout so that the status is already 'cancelled' by the time the
        API returns — preventing the race where the client polls /train/status and
        still sees 'training' after a successful cancel call.
        """
        with self._lock:
            current_status = self.state.get("status")
            if current_status in ["cancelling", "cancelled"]:
                return True
            if current_status not in ["syncing", "training", "evaluating"]:
                return False
            self.state["status"] = "cancelling"
            self.state["message"] = "Stopping training..."
            self._stop_requested.set()

        self.log("⚠ Cancellation requested by user...")

        # Wait outside the lock so the worker thread can acquire it to write its
        # final 'cancelled' state without deadlocking against cancel_training.
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=wait_seconds)

        return True

    def _evaluate_baseline(self, model: nn.Module, val_loader, criterion, device) -> tuple:
        """Evaluate baseline accuracy before training begins."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                if self._stop_requested.is_set():
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
    ):
        """Worker thread executing the training loop with Validation Guard."""
        try:
            # Determine if retraining full ensemble or a single backbone
            if architecture in ["ensemble", "all", "tri_model"]:
                arch_list = ["swin_transformer", "resnet50", "efficientnet_v2"]
                is_ensemble_run = True
            else:
                arch_list = [architecture]
                is_ensemble_run = False

            total_models = len(arch_list)
            self.log("=" * 64)
            self.log("   DERMASSIST - MULTI-MODEL BENCHMARK TRAINING PIPELINE")
            self.log("=" * 64)
            self.log(f"   Architecture Pipeline : {'TRI-MODEL ENSEMBLE' if is_ensemble_run else arch_list[0].upper()}")
            self.log(f"   Active Models         : {', '.join([a.upper() for a in arch_list])}")
            self.log(f"   Epochs Per Backbone   : {epochs} epoch(s)")
            self.log(f"   Dataset Auto-Sync     : {'Enabled' if sync_dataset else 'Disabled'}")
            self.log("=" * 64)

            # Step 1: Sync dataset if requested
            if sync_dataset:
                self.sync_gathered_dataset()

            if self._stop_requested.is_set():
                self._handle_cancellation()
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.log(f"Compute hardware device : {str(device).upper()} (PyTorch {torch.__version__})")

            promoted_models = []
            results_summary = []
            smoothed_step_duration: Optional[float] = None
            avg_val_duration: float = 3.0

            for arch_idx, arch in enumerate(arch_list):
                if self._stop_requested.is_set():
                    self._handle_cancellation()
                    return

                model_num = arch_idx + 1
                self.log("-" * 64)
                self.log(f"▶ [{model_num}/{total_models}] TRAINING BACKBONE: {arch.upper()}")
                self.log("-" * 64)

                with self._lock:
                    self.state["status"] = "training"
                    self.state["architecture"] = arch
                    self.state["message"] = (
                        f"Training Model {model_num}/{total_models} ({arch.replace('_', ' ').title()})..."
                        if is_ensemble_run
                        else f"Training {arch.replace('_', ' ').title()}..."
                    )

                config = load_config(self.config_path)
                config["advanced"]["active_architecture"] = arch
                config["training"]["epochs"] = epochs
                if learning_rate is not None:
                    config["training"]["learning_rate"] = learning_rate

                # Step 2: Dataloaders
                self.log(f"Building clinical data loaders for {arch.upper()}...")
                train_loader, val_loader, class_names = create_dataloaders(config)
                total_batches = len(train_loader)

                with self._lock:
                    self.state["total_batches"] = total_batches

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
                    with self._lock:
                        self.state["status"] = "evaluating"
                        self.state["message"] = f"Measuring baseline for {arch.replace('_', ' ').title()}..."
                    _, baseline_acc = self._evaluate_baseline(model, val_loader, criterion, device)
                    if self._stop_requested.is_set():
                        self._handle_cancellation()
                        return
                    self.log(f"Measured baseline accuracy: {baseline_acc:.2f}%")

                with self._lock:
                    self.state["baseline_val_acc"] = round(baseline_acc, 2)
                    self.state["status"] = "training"

                # Setup Optimizer & Scheduler
                lr = learning_rate or (0.00005 if arch == "swin_transformer" else 0.0001)
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

                best_val_acc = baseline_acc
                new_best_model_weights = None
                total_steps = epochs * total_batches
                step_count = 0

                # Training Epochs
                for epoch in range(epochs):
                    if self._stop_requested.is_set():
                        self._handle_cancellation()
                        return

                    epoch_start_time = time.time()
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    with self._lock:
                        self.state["current_epoch"] = epoch + 1
                        self.state["message"] = (
                            f"[{model_num}/{total_models}] {arch.replace('_', ' ').title()} - Epoch {epoch + 1}/{epochs}"
                            if is_ensemble_run
                            else f"Training Epoch {epoch + 1}/{epochs}"
                        )

                    for batch_idx, (images, labels) in enumerate(train_loader):
                        if self._stop_requested.is_set():
                            self._handle_cancellation()
                            return

                        batch_start_time = time.time()
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        batch_duration = max(time.time() - batch_start_time, 0.001)
                        if smoothed_step_duration is None:
                            smoothed_step_duration = batch_duration
                        else:
                            smoothed_step_duration = (0.85 * smoothed_step_duration) + (0.15 * batch_duration)

                        running_loss += loss.item() * images.size(0)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        step_count += 1

                        # Periodic progress update & live terminal bar
                        if batch_idx % 2 == 0 or batch_idx == total_batches - 1:
                            sub_progress = (step_count / max(total_steps, 1)) * 100.0
                            overall_progress = round(
                                ((arch_idx * 100.0) + sub_progress) / total_models, 1
                            )
                            current_train_acc = 100.0 * correct / max(total, 1)
                            current_train_loss = running_loss / max(total, 1)

                            total_estimated_steps = total_steps * total_models
                            cumulative_steps = (arch_idx * total_steps) + step_count
                            remaining_steps = max(0, total_estimated_steps - cumulative_steps)
                            remaining_epochs = max(0, ((total_models - arch_idx - 1) * epochs) + (epochs - (epoch + 1)))
                            step_time = smoothed_step_duration if smoothed_step_duration is not None else batch_duration
                            eta = int((remaining_steps * step_time) + (remaining_epochs * avg_val_duration))

                            with self._lock:
                                self.state["progress"] = min(overall_progress, 99.0)
                                self.state["current_batch"] = batch_idx + 1
                                self.state["train_loss"] = round(current_train_loss, 4)
                                self.state["train_acc"] = round(current_train_acc, 2)
                                self.state["eta_seconds"] = eta

                            # Generate ASCII progress bar for live terminal stream
                            pct = int((batch_idx + 1) / max(total_batches, 1) * 100)
                            bar_len = 16
                            filled_len = int(bar_len * (batch_idx + 1) / max(total_batches, 1))
                            bar_str = "=" * max(0, filled_len - 1) + (">" if filled_len > 0 and filled_len < bar_len else ("=" if filled_len == bar_len else ""))
                            bar_str = bar_str.ljust(bar_len, ".")

                            progress_log = (
                                f"[STEP] Epoch {epoch + 1:2d}/{epochs:2d} [{bar_str}] {pct:3d}% "
                                f"({batch_idx + 1}/{total_batches}) | "
                                f"Loss: {current_train_loss:.4f} | Acc: {current_train_acc:5.2f}% | "
                                f"Speed: {batch_duration:.2f}s/step | ETA: {eta}s"
                            )
                            # Update terminal line in-place for active step, or append when done with epoch
                            is_epoch_done = (batch_idx == total_batches - 1)
                            self.log(progress_log, replace_last=(batch_idx > 0 and not is_epoch_done))

                    # Validation Phase
                    scheduler.step()
                    model.eval()
                    val_start_time = time.time()
                    val_loss_sum = 0.0
                    val_correct = 0
                    val_total = 0

                    with self._lock:
                        self.state["status"] = "evaluating"
                        self.state["message"] = f"Validating {arch.replace('_', ' ').title()} (Epoch {epoch + 1}/{epochs})..."

                    with torch.no_grad():
                        for images, labels in val_loader:
                            # Bug fix: check for cancellation inside the validation loop
                            # so the thread doesn't keep running through a full val pass
                            # after the user has already pressed cancel.
                            if self._stop_requested.is_set():
                                break

                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_loss_sum += loss.item() * images.size(0)
                            _, predicted = outputs.max(1)
                            val_total += labels.size(0)
                            val_correct += predicted.eq(labels).sum().item()

                    # Propagate cancellation after breaking out of the val loop
                    if self._stop_requested.is_set():
                        self._handle_cancellation()
                        return

                    val_duration = max(time.time() - val_start_time, 0.5)
                    avg_val_duration = (0.7 * avg_val_duration) + (0.3 * val_duration)

                    epoch_train_loss = running_loss / max(total, 1)
                    epoch_train_acc = 100.0 * correct / max(total, 1)
                    epoch_val_loss = val_loss_sum / max(val_total, 1)
                    epoch_val_acc = 100.0 * val_correct / max(val_total, 1)

                    is_best = epoch_val_acc > best_val_acc
                    if is_best:
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
                        + (" ⭐ (New Best)" if is_best else "")
                    )

                    with self._lock:
                        self.state["status"] = "training"
                        self.state["val_loss"] = round(epoch_val_loss, 4)
                        self.state["val_acc"] = round(epoch_val_acc, 2)
                        self.state["best_val_acc"] = round(best_val_acc, 2)
                        self.state["history"]["train_loss"].append(round(epoch_train_loss, 4))
                        self.state["history"]["train_acc"].append(round(epoch_train_acc, 2))
                        self.state["history"]["val_loss"].append(round(epoch_val_loss, 4))
                        self.state["history"]["val_acc"].append(round(epoch_val_acc, 2))

                # Step 4: VALIDATION GUARD CHECK FOR THIS MODEL
                self.log(f"--- VALIDATION GUARD: {arch.upper()} ---")
                self.log(f"Baseline: {baseline_acc:.2f}% | Best Achieved: {best_val_acc:.2f}%")

                target_prod_path = os.path.join("models/production", f"best_model_{arch}.pth")
                backup_prod_path = os.path.join("models/production", f"best_model_{arch}_backup.pth")
                os.makedirs("models/production", exist_ok=True)

                if new_best_model_weights is not None and best_val_acc >= baseline_acc:
                    self.log(f"✅ PASSED GUARD: {arch.upper()} improved ({baseline_acc:.2f}% -> {best_val_acc:.2f}%).")
                    if os.path.exists(target_prod_path):
                        shutil.copy2(target_prod_path, backup_prod_path)
                    torch.save(new_best_model_weights, target_prod_path)
                    self.log(f"🚀 Deployed to production -> {target_prod_path}")
                    promoted_models.append(arch)
                    results_summary.append(f"{arch.upper()}: {best_val_acc:.2f}% (Promoted)")
                else:
                    self.log(f"🛡 GUARD PRESERVED: {arch.upper()} did not beat baseline ({baseline_acc:.2f}%).")
                    results_summary.append(f"{arch.upper()}: Baseline {baseline_acc:.2f}% kept")

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
                self.state["status"] = "completed"
                self.state["progress"] = 100.0
                self.state["eta_seconds"] = 0
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
            import traceback
            err_msg = traceback.format_exc()
            self.log(f"❌ Training failed with exception: {e}")
            print(err_msg)
            with self._lock:
                self.state["status"] = "failed"
                self.state["message"] = f"Training failed: {str(e)}"
                self.state["eta_seconds"] = 0
                self.state["completed_at"] = datetime.now().isoformat()

    def _handle_cancellation(self):
        """Clean up when cancellation is requested."""
        self.log("🛑 Training successfully cancelled by user.")
        with self._lock:
            self.state["status"] = "cancelled"
            self.state["message"] = "Training was cancelled."
            self.state["eta_seconds"] = 0
            self.state["completed_at"] = datetime.now().isoformat()


# Global Singleton
training_manager = TrainingManager()
