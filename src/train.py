"""
DermAssist - Training Script
==============================
Full training loop with validation, learning rate scheduling,
early stopping, and model checkpointing.

Usage:
    python -m src.train
    python -m src.train --config config.yaml
    python -m src.train --resume
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_config, create_dataloaders
from src.model import build_model


# ============================================================
# Training Engine
# ============================================================

class Trainer:
    """Encapsulates the training and validation loop."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("-" * 60)
        print("  DermAssist - Skin Lesion Detection Training")
        print("-" * 60)
        print(f"  Device: {self.device}")
        print(f"  Epochs: {config['training']['epochs']}")
        print(f"  Batch:  {config['training']['batch_size']}")
        print(f"  LR:     {config['training']['learning_rate']}")
        print("-" * 60)

        # ---- Data ----
        self.train_loader, self.val_loader, self.class_names = \
            create_dataloaders(config)

        # ---- Model ----
        self.model = build_model(config, self.device)

        # ---- Loss & Optimizer ----
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # ---- Scheduler ----
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config["training"]["lr_step_size"],
            gamma=config["training"]["lr_gamma"],
        )

        # ---- Tracking ----
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        # ---- Directories ----
        Path(config["training"]["checkpoint_dir"]).mkdir(
            parents=True, exist_ok=True
        )
        Path(config["training"]["production_model_path"]).parent.mkdir(
            parents=True, exist_ok=True
        )

    # ---------------------------------------------------------
    # Resume from checkpoint
    # ---------------------------------------------------------
    def load_checkpoint(self, checkpoint_path: str = None):
        """Load model, optimizer, and scheduler state from a checkpoint."""
        if checkpoint_path is None:
            # Try to find the latest checkpoint in the checkpoint_dir
            ckpt_dir = self.config["training"]["checkpoint_dir"]
            ckpts = list(Path(ckpt_dir).glob("checkpoint_epoch_*.pth"))
            if not ckpts:
                print("  ! No checkpoints found to resume. Starting from scratch.")
                return
            # Sort by epoch number
            ckpts.sort(key=lambda x: int(x.stem.split("_")[-1]))
            checkpoint_path = str(ckpts[-1])

        print(f"  -> Resuming from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_acc = checkpoint.get("val_acc", 0.0)
        
        print(f"  -> Restarting from Epoch {self.start_epoch}")

    # ---------------------------------------------------------
    # Train one epoch
    # ---------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> tuple:
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1:3d} [Train] ",
            leave=False,
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Stats
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------
    def validate(self, epoch: int) -> tuple:
        """Run validation. Returns (avg_loss, accuracy)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1} [Val]  ",
            leave=False,
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ---------------------------------------------------------
    # Save checkpoint
    # ---------------------------------------------------------
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool):
        """Save model checkpoint and optionally the best model."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "class_names": self.class_names,
            "config": self.config,
            "architecture": self.config["advanced"].get("active_architecture", "resnet50"),
        }

        # Save periodic checkpoint
        ckpt_path = os.path.join(
            self.config["training"]["checkpoint_dir"],
            f"checkpoint_epoch_{epoch + 1}.pth",
        )
        torch.save(checkpoint, ckpt_path)

        # Save best model for production
        if is_best:
            arch_name = self.config["advanced"].get("active_architecture", "resnet50")
            prod_path = os.path.join(
                os.path.dirname(self.config["training"]["production_model_path"]),
                f"best_model_{arch_name}.pth"
            )
            torch.save(checkpoint, prod_path)
            print(f"  * Best model saved -> {prod_path} "
                  f"(val_acc={val_acc:.2f}%)")

    # ---------------------------------------------------------
    # Plot training history
    # ---------------------------------------------------------
    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs_range = range(1, len(self.history["train_loss"]) + 1)

        # Loss
        ax1.plot(epochs_range, self.history["train_loss"], label="Train Loss")
        ax1.plot(epochs_range, self.history["val_loss"], label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs_range, self.history["train_acc"], label="Train Acc")
        ax2.plot(epochs_range, self.history["val_acc"], label="Val Acc")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150)
        plt.close()
        print("[Train] Training curves saved → training_history.png")

    # ---------------------------------------------------------
    # Full training loop
    # ---------------------------------------------------------
    def train(self, epochs_override: Optional[int] = None):
        """Execute the full training loop."""
        total_epochs = epochs_override if epochs_override is not None else self.config["training"]["epochs"]
        patience = self.config["training"]["early_stopping_patience"]

        print(f"\n{'-' * 60}")
        print(f"  Training from Epoch {self.start_epoch + 1} to {total_epochs}...")
        print(f"{'-' * 60}\n")

        start_time = time.time()

        for epoch in range(self.start_epoch, total_epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate(epoch)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Print epoch summary
            print(
                f"  Epoch {epoch + 1:3d}/{total_epochs} | "
                f"Train Loss: {train_loss:.4f}  Acc: {train_acc:6.2f}% | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:6.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            # Check for improvement
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_acc, is_best)

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n  ! Early stopping triggered after {epoch + 1} epochs")
                break

        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print(f"\n{'=' * 60}")
        print(f"  Training Complete!")
        print(f"  Duration:       {minutes}m {seconds}s")
        print(f"  Best Val Acc:   {self.best_val_acc:.2f}%")
        print(f"{'=' * 60}\n")
        self.plot_history()
        return self.history


# ============================================================
# Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DermAssist — Train skin lesion classifier"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Specific checkpoint path to resume from"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        help="Override architecture (resnet50, efficientnet_v2, swin_transformer)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Override output model filename"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply command-line overrides to config
    if args.arch:
        if "advanced" not in config:
            config["advanced"] = {}
        config["advanced"]["active_architecture"] = args.arch
        print(f"[Train] Architecture override: {args.arch}")

    if args.name:
        # Update the production model path filename
        prod_path = config["training"]["production_model_path"]
        prod_dir = os.path.dirname(prod_path)
        config["training"]["production_model_path"] = os.path.join(prod_dir, args.name)
        print(f"[Train] Model name override: {args.name}")
    trainer = Trainer(config)
    
    if args.resume or args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    trainer.train(epochs_override=args.epochs)


if __name__ == "__main__":
    main()
