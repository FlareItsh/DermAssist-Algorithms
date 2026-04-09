"""
DermAssist — HAM10000 Dataset Organizer
==========================================
Reads the HAM10000 metadata CSV and sorts images from the flat
image folders into class subdirectories expected by data_loader.py.

Usage:
    python organize_dataset.py --csv path/to/HAM10000_metadata.csv --images path/to/images

Example:
    python organize_dataset.py \
        --csv  "C:/Users/MYPC1/Downloads/HAM10000_metadata.csv" \
        --images "C:/Users/MYPC1/Downloads/HAM10000_images_part_1" "C:/Users/MYPC1/Downloads/HAM10000_images_part_2" \
        --output "data/raw"
"""

import os
import csv
import shutil
import argparse
from pathlib import Path
from collections import Counter


# ============================================================
# Diagnosis code → human-readable class name
# ============================================================
DIAGNOSIS_MAP = {
    "akiec": "Actinic keratoses",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}


def organize_dataset(csv_path: str, image_dirs: list, output_dir: str, copy: bool = True):
    """
    Read the HAM10000 metadata CSV and sort images into
    class subdirectories.

    Args:
        csv_path:   Path to HAM10000_metadata.csv
        image_dirs: List of paths to image folders (part_1, part_2, etc.)
        output_dir: Destination directory (e.g. data/raw/)
        copy:       If True, copy files. If False, move files.
    """
    output_path = Path(output_dir)

    # ---- Step 1: Create class subdirectories ----
    print("=" * 60)
    print("  DermAssist — Dataset Organizer")
    print("=" * 60)
    print(f"  CSV:     {csv_path}")
    print(f"  Images:  {image_dirs}")
    print(f"  Output:  {output_dir}")
    print(f"  Mode:    {'COPY' if copy else 'MOVE'}")
    print("=" * 60)

    for class_name in DIAGNOSIS_MAP.values():
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 2: Build a lookup of all available images ----
    print("\n[1/3] Scanning image directories...")
    image_lookup = {}
    for img_dir in image_dirs:
        img_path = Path(img_dir)
        if not img_path.is_dir():
            print(f"  ⚠ Directory not found: {img_dir}")
            continue
        for img_file in img_path.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                # Key: image ID without extension (e.g. "ISIC_0029306")
                image_lookup[img_file.stem] = img_file
    print(f"  Found {len(image_lookup)} images")

    # ---- Step 3: Read CSV and sort images ----
    print("\n[2/3] Reading metadata CSV and organizing images...")
    stats = Counter()
    not_found = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = row["image_id"]        # e.g. "ISIC_0029306"
            diagnosis = row["dx"]              # e.g. "mel"

            # Map diagnosis code to class name
            class_name = DIAGNOSIS_MAP.get(diagnosis)
            if class_name is None:
                print(f"  ⚠ Unknown diagnosis '{diagnosis}' for {image_id}, skipping")
                continue

            # Find the image file
            if image_id not in image_lookup:
                not_found.append(image_id)
                continue

            src = image_lookup[image_id]
            dst = output_path / class_name / src.name

            # Copy or move
            if not dst.exists():
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(str(src), str(dst))
                stats[class_name] += 1

    # ---- Step 4: Summary ----
    print("\n[3/3] Done! Summary:")
    print(f"{'─' * 50}")
    total = 0
    for class_name in sorted(DIAGNOSIS_MAP.values()):
        count = stats[class_name]
        total += count
        bar = "█" * (count // 50)
        print(f"  {class_name:30s} {count:5d}  {bar}")
    print(f"{'─' * 50}")
    print(f"  {'TOTAL':30s} {total:5d}")

    if not_found:
        print(f"\n  ⚠ {len(not_found)} images in CSV were not found in image directories")
        if len(not_found) <= 10:
            for nf in not_found:
                print(f"    - {nf}")
        else:
            for nf in not_found[:5]:
                print(f"    - {nf}")
            print(f"    ... and {len(not_found) - 5} more")

    print(f"\n✅ Dataset organized in: {output_path.resolve()}")
    print(f"   You can now run: python -m src.train")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize HAM10000 images into class subdirectories"
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to HAM10000_metadata.csv"
    )
    parser.add_argument(
        "--images", type=str, nargs="+", required=True,
        help="Path(s) to image folders (part_1, part_2, etc.)"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move files instead of copying (saves disk space)"
    )

    args = parser.parse_args()

    organize_dataset(
        csv_path=args.csv,
        image_dirs=args.images,
        output_dir=args.output,
        copy=not args.move,
    )
