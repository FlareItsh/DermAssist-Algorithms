"""
DermAssist - Build Locked Holdout Test Set
==========================================
Extracts a clean, non-leaking holdout test set from data/raw/ into data/holdout_test/
and archives corresponding augmented copies (aug_*) into data/holdout_aug_archive/.

Option C (Hybrid Strategy):
- Prioritizes real webapp clinical uploads (unseen by original baseline models).
- Complements with distinct original base images (up to target quotas).
- Strictly preserves reversibility with a complete manifest.json and --restore flag.
- Dry-run by default to prevent accidental data changes.
"""

import os
import sys
import json
import shutil
import random
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.data_loader import load_config, get_image_group, VALID_IMAGE_EXTENSIONS


def is_training_active(port: int = 8001) -> bool:
    """Check if the training manager is currently running an active job."""
    url = f"http://127.0.0.1:{port}/train/status"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HoldoutBuilder/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status = data.get("status", "idle")
            return status in ["syncing", "training", "evaluating", "cancelling"]
    except Exception:
        # If server is not responding, training thread is definitely not running
        return False


def build_holdout(
    config_path: str = "config.yaml",
    target_counts: Dict[str, int] = None,
    seed: int = 42,
    apply: bool = False,
):
    if target_counts is None:
        target_counts = {"Acne": 100, "Eczema": 100, "Herpes": 60}

    config = load_config(config_path)
    raw_dir = config["data"]["raw_dir"]
    holdout_dir = config["data"].get("holdout_test_dir", "data/holdout_test")
    archive_dir = "data/holdout_aug_archive"

    print("=" * 65)
    print("DERMASSIST: LOCKED HOLDOUT TEST SET GENERATOR")
    print(f"Mode: {'APPLY (Moving Files)' if apply else 'DRY RUN (Preview Only)'}")
    print(f"Target counts: {target_counts}")
    print(f"Seed: {seed}")
    print("=" * 65)

    if apply and is_training_active(config.get("api", {}).get("port", 8001)):
        print("\n❌ ERROR: Model retraining is currently active!")
        print("Cannot modify dataset while a training pipeline is in progress.")
        sys.exit(1)

    rng = random.Random(seed)
    manifest = {
        "created_at": None,
        "seed": seed,
        "target_counts": target_counts,
        "holdout_files": [],  # List of {class, original_path, target_path, is_webapp}
        "archived_augs": [],   # List of {class, original_path, target_path, group}
    }

    for class_name, target_count in target_counts.items():
        class_raw_dir = os.path.join(raw_dir, class_name)
        if not os.path.exists(class_raw_dir):
            print(f"⚠ Warning: Class directory not found: {class_raw_dir}")
            continue

        # 1. Scan files in class directory
        all_files = [
            f for f in os.listdir(class_raw_dir)
            if os.path.isfile(os.path.join(class_raw_dir, f))
            and os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
        ]

        # Categorize
        webapp_files = [f for f in all_files if f.startswith("webapp_")]
        aug_files = [f for f in all_files if f.startswith("aug_")]
        base_originals = [f for f in all_files if not f.startswith("webapp_") and not f.startswith("aug_")]

        # Group augmented files by their original source name
        augs_by_group: Dict[str, List[str]] = {}
        for af in aug_files:
            group = get_image_group(af)
            augs_by_group.setdefault(group, []).append(af)

        print(f"\n[{class_name}] Found:")
        print(f"  • Webapp uploads : {len(webapp_files)}")
        print(f"  • Base originals : {len(base_originals)}")
        print(f"  • Augmented files: {len(aug_files)} across {len(augs_by_group)} source groups")

        # 2. Select candidates (Hybrid strategy: all webapp uploads first, then base originals)
        selected_groups = set()
        chosen_holdout_files = []

        # A: Include webapp files
        for wf in sorted(webapp_files):
            chosen_holdout_files.append((wf, True))
            selected_groups.add(get_image_group(wf))

        # B: Sample remaining needed from base originals
        remaining_needed = max(0, target_count - len(chosen_holdout_files))
        shuffled_base = list(sorted(base_originals))
        rng.shuffle(shuffled_base)

        for bf in shuffled_base:
            if len(chosen_holdout_files) >= target_count:
                break
            group = get_image_group(bf)
            if group not in selected_groups:
                selected_groups.add(group)
                chosen_holdout_files.append((bf, False))

        print(f"  -> Selected for Holdout: {len(chosen_holdout_files)} images "
              f"({sum(1 for _, w in chosen_holdout_files if w)} webapp + "
              f"{sum(1 for _, w in chosen_holdout_files if not w)} base originals)")

        # 3. Identify all corresponding aug_ files that MUST be archived to prevent leakage
        matching_augs = []
        for group in selected_groups:
            if group in augs_by_group:
                matching_augs.extend(augs_by_group[group])

        print(f"  -> Augmented copies to archive: {len(matching_augs)} files")

        # Record into manifest
        class_holdout_dir = os.path.join(holdout_dir, class_name)
        class_archive_dir = os.path.join(archive_dir, class_name)

        for fname, is_wa in chosen_holdout_files:
            manifest["holdout_files"].append({
                "class": class_name,
                "filename": fname,
                "source_path": os.path.join(class_raw_dir, fname),
                "dest_path": os.path.join(class_holdout_dir, fname),
                "is_webapp": is_wa,
                "group": get_image_group(fname),
            })

        for afname in matching_augs:
            manifest["archived_augs"].append({
                "class": class_name,
                "filename": afname,
                "source_path": os.path.join(class_raw_dir, afname),
                "dest_path": os.path.join(class_archive_dir, afname),
                "group": get_image_group(afname),
            })

    total_holdout = len(manifest["holdout_files"])
    total_archived = len(manifest["archived_augs"])
    print("\n" + "=" * 65)
    print("SUMMARY:")
    print(f"  Total holdout test images : {total_holdout}")
    print(f"  Total augmented copies to archive: {total_archived}")
    print("=" * 65)

    if not apply:
        print("\n[INFO] This was a DRY RUN. No files were modified.")
        print("To execute this change, run:")
        print("    python src/build_holdout_set.py --apply")
        return

    # Apply changes
    from datetime import datetime
    manifest["created_at"] = datetime.now().isoformat()

    print("\n[Executing File Operations...]")
    for item in manifest["holdout_files"]:
        os.makedirs(os.path.dirname(item["dest_path"]), exist_ok=True)
        shutil.move(item["source_path"], item["dest_path"])

    for item in manifest["archived_augs"]:
        os.makedirs(os.path.dirname(item["dest_path"]), exist_ok=True)
        shutil.move(item["source_path"], item["dest_path"])

    # Write manifest
    os.makedirs(holdout_dir, exist_ok=True)
    manifest_path = os.path.join(holdout_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Holdout set successfully created in: {holdout_dir}")
    print(f"[OK] Augmented copies safely archived in: {archive_dir}")
    print(f"[OK] Manifest recorded at: {manifest_path}")
    print("All models will now be scored fairly against this locked test set!")


def restore_holdout(config_path: str = "config.yaml"):
    config = load_config(config_path)
    holdout_dir = config["data"].get("holdout_test_dir", "data/holdout_test")
    archive_dir = "data/holdout_aug_archive"
    manifest_path = os.path.join(holdout_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest file not found at: {manifest_path}")
        print("Cannot restore without a manifest.")
        sys.exit(1)

    if is_training_active(config.get("api", {}).get("port", 8001)):
        print("\n[ERROR] Model retraining is currently active!")
        print("Cannot restore dataset while a training pipeline is in progress.")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 65)
    print("DERMASSIST: RESTORING HOLDOUT SET BACK TO DATA/RAW")
    print(f"Holdout files to restore: {len(manifest.get('holdout_files', []))}")
    print(f"Archived augs to restore: {len(manifest.get('archived_augs', []))}")
    print("=" * 65)

    restored_holdout = 0
    restored_augs = 0

    # 1. Restore holdout files
    for item in manifest.get("holdout_files", []):
        current_loc = item["dest_path"]
        orig_loc = item["source_path"]
        if os.path.exists(current_loc):
            os.makedirs(os.path.dirname(orig_loc), exist_ok=True)
            shutil.move(current_loc, orig_loc)
            restored_holdout += 1

    # 2. Restore aug files
    for item in manifest.get("archived_augs", []):
        current_loc = item["dest_path"]
        orig_loc = item["source_path"]
        if os.path.exists(current_loc):
            os.makedirs(os.path.dirname(orig_loc), exist_ok=True)
            shutil.move(current_loc, orig_loc)
            restored_augs += 1

    # 3. Clean up holdout and archive folders
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    for base in [holdout_dir, archive_dir]:
        if os.path.exists(base):
            shutil.rmtree(base, ignore_errors=True)

    print(f"[OK] Successfully restored {restored_holdout} holdout files and {restored_augs} aug files.")
    print("[OK] data/raw/ restored to previous state and holdout manifests cleared.")


def main():
    parser = argparse.ArgumentParser(description="Build or restore locked holdout test set.")
    parser.add_argument("--apply", action="store_true", help="Execute file moves (dry run if omitted)")
    parser.add_argument("--restore", action="store_true", help="Restore all moved files back to data/raw/")
    parser.add_argument("--acne", type=int, default=100, help="Target Acne count (default: 100)")
    parser.add_argument("--eczema", type=int, default=100, help="Target Eczema count (default: 100)")
    parser.add_argument("--herpes", type=int, default=60, help="Target Herpes count (default: 60)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic selection")

    args = parser.parse_args()

    if args.restore:
        restore_holdout()
    else:
        target_counts = {"Acne": args.acne, "Eczema": args.eczema, "Herpes": args.herpes}
        build_holdout(target_counts=target_counts, seed=args.seed, apply=args.apply)


if __name__ == "__main__":
    main()
