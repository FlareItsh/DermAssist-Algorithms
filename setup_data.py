import os
import shutil
import argparse
import kagglehub
from pathlib import Path

def setup_dermassist_data(local_path=None):
    print("🚀 Starting DermAssist Data Setup...")
    
    if local_path:
        print(f"📁 Using local dataset path: {local_path}")
        source_root = Path(local_path)
    else:
        # 1. Download latest version of the dataset
        print("📥 Downloading dataset from Kaggle (this may take a while)...")
        cache_path = kagglehub.dataset_download("muhammadabdulsami/massive-skin-disease-balanced-dataset")
        print(f"✅ Downloaded to: {cache_path}")
        source_root = Path(cache_path)

    # Some Kaggle datasets have nested folders, we try to find the 'balanced_dataset' root
    potential_roots = [
        source_root / "balanced_dataset" / "balanced_dataset",
        source_root / "balanced_dataset",
        source_root
    ]
    
    actual_source = None
    for p in potential_roots:
        if (p / "Acne And Rosacea Photos").exists():
            actual_source = p
            break
            
    if not actual_source:
        print(f"❌ Could not find skin disease folders in: {source_root}")
        return

    dest_root = Path("data/raw")
    dest_root.mkdir(parents=True, exist_ok=True)

    # Mapping: Kaggle Folder -> Our Folder
    class_mapping = {
        "Acne And Rosacea Photos": "Acne",
        "Eczema Photos": "Eczema",
        "Herpes HPV and other STDs Photos": "Herpes"
    }

    # 3. Copy only the required folders
    print(f"\n📦 Extracting specific classes from {actual_source}...")
    for kaggle_folder, our_folder in class_mapping.items():
        src = actual_source / kaggle_folder
        dst = dest_root / our_folder
        
        if src.exists():
            print(f"  - Copying {kaggle_folder} -> {our_folder}...")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            print(f"  ❌ Source folder not found: {src}")

    print(f"\n✅ Data setup complete! Folders ready in: {dest_root.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup DermAssist data from Kaggle or local path")
    parser.add_argument("--path", type=str, help="Local path to the unzipped dataset")
    args = parser.parse_args()
    
    setup_dermassist_data(args.path)
