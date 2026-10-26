"""
DermAssist - Repository Path Resolver for Python/Algorithms
===========================================================
Canonically resolves paths between algorithms and web API storage.
"""

import os
from pathlib import Path
from typing import Optional


def get_algorithms_root() -> Path:
    """Return the absolute path to the algorithms root directory."""
    return Path(__file__).resolve().parent.parent


def get_algorithms_raw_dir() -> Path:
    """Return the absolute path to algorithms/data/raw."""
    env_dir = os.environ.get("DERMASSIST_ALGORITHMS_DATA_DIR")
    if env_dir and os.path.isdir(env_dir):
        return Path(env_dir)

    default_path = get_algorithms_root() / "data" / "raw"
    default_path.mkdir(parents=True, exist_ok=True)
    return default_path


def get_api_storage_public_dir() -> Optional[Path]:
    """Return the absolute path to api/storage/app/public if available."""
    env_dir = os.environ.get("DERMASSIST_STORAGE_PUBLIC_DIR")
    if env_dir and os.path.isdir(env_dir):
        return Path(env_dir)

    candidates = [
        get_algorithms_root().parent / "api" / "storage" / "app" / "public",
        Path.cwd().parent / "api" / "storage" / "app" / "public",
        get_algorithms_root().parent / "DermAssist-API" / "storage" / "app" / "public",
    ]
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    return None


def get_api_storage_dataset_dir() -> Optional[Path]:
    """Return the absolute path to api/storage/app/public/dataset if available."""
    env_dir = os.environ.get("DERMASSIST_DATASET_DIR")
    if env_dir and os.path.isdir(env_dir):
        return Path(env_dir)

    public_dir = get_api_storage_public_dir()
    if public_dir:
        dataset_dir = public_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    return None
