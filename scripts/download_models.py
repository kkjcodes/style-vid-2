"""
Pre-download all model weights to the HuggingFace cache.
Run once before starting the server:
  PYTHONPATH=. python scripts/download_models.py

Downloads:
  - Wan2.1-I2V-14B-480P-Diffusers  (~30GB)
  - InsightFace buffalo_l            (~300MB, auto on first use)

Wan2.2-Animate-14B is ~55GB — only download if using Track B.
  PYTHONPATH=. python scripts/download_models.py --animate
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import snapshot_download
from backend.core.config import get_settings

settings = get_settings()


def download_i2v():
    print(f"[download] Downloading {settings.wan_i2v_model_id} (~30GB)…")
    snapshot_download(repo_id=settings.wan_i2v_model_id)
    print("[download] I2V model ready.")


def download_animate():
    print(f"[download] Downloading {settings.wan_animate_model_id} (~55GB)…")
    snapshot_download(repo_id=settings.wan_animate_model_id)
    print("[download] Animate model ready.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animate", action="store_true", help="Also download Wan2.2-Animate (55GB)")
    args = parser.parse_args()

    download_i2v()
    if args.animate:
        download_animate()

    print("\n[download] All done. Run: docker compose up -d && uvicorn backend.api.main:app --reload")


if __name__ == "__main__":
    main()
