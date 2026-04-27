"""
DermAssist - Automated Multi-Model Training
===========================================
This script automatically trains all three planned architectures:
1. ResNet50 (Legacy/Stable)
2. EfficientNet-V2 (Improved)
3. Swin Transformer (State-of-the-art)

Usage:
    python train_all.py           (Start from scratch)
    python train_all.py --resume  (Resume from last checkpoint)
"""

import subprocess
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="DermAssist - Automated Benchmark Trainer")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per model")
    args = parser.parse_args()

    architectures = [
        "resnet50",
        "efficientnet_v2",
        "swin_transformer"
    ]
    
    print("\n" + "="*60)
    print(" DERMASSIST - AUTOMATED BENCHMARK PIPELINE")
    print("="*60)
    print(f" Models to train: {len(architectures)}")
    print(f" Epochs per model: {args.epochs}")
    print(f" Mode:             {'RESUME' if args.resume else 'FRESH START'}")
    print("="*60 + "\n")

    for i, arch in enumerate(architectures):
        print(f"[{i+1}/{len(architectures)}] Current Architecture: {arch.upper()}")
        
        # Check if the final model already exists
        model_final_path = f"models/production/best_model_{arch}.pth"
        if os.path.exists(model_final_path) and not args.resume:
            print(f" ! Final model already exists at {model_final_path}. Skipping.")
            continue

        # Construct the command for src.train
        # Note: src.train handles the internal config updates per architecture
        cmd = [
            sys.executable, "-m", "src.train",
            "--arch", arch,
            "--epochs", str(args.epochs),
            "--name", f"best_model_{arch}.pth"
        ]
        
        if args.resume:
            cmd.append("--resume")
            
        print(f" -> Executing: {' '.join(cmd)}")
        
        try:
            # We use subprocess.run so we wait for each model to finish
            subprocess.run(cmd, check=True)
            print(f" ✅ SUCCESS: Finished {arch.upper()}\n")
        except subprocess.CalledProcessError:
            print(f" ❌ ERROR: Training failed for {arch}. Stopping pipeline.")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n ⚠ Training interrupted by user. You can resume later with --resume.")
            sys.exit(0)

    print("="*60)
    print(" 🎉 ALL MODELS TRAINED AND SAVED TO 'models/production/'")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
