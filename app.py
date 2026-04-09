"""
DermAssist — Skin Lesion Detection
====================================
Entry point script. Use this to quickly launch the API server.

Usage:
    python app.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Launch the DermAssist API server."""
    print("🩺 DermAssist — Skin Lesion Detection")
    print("=" * 45)
    print()
    print("Available commands:")
    print("  1. Train model:     python -m src.train")
    print("  2. Run inference:   python -m src.inference --image <path>")
    print("  3. Start API:       python api/app.py")
    print("  4. Start API (alt): uvicorn api.app:app --reload")
    print()

    # Default: start the API server
    from api.app import app
    import uvicorn
    import yaml

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    host = config["api"]["host"]
    port = config["api"]["port"]

    print(f"🚀 Starting API server on http://{host}:{port}")
    print(f"📖 Docs at http://{host}:{port}/docs")
    print()

    uvicorn.run("api.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()