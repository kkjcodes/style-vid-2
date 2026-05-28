"""Lightweight selfie face detection for API-side validation.

This module intentionally avoids loading large face-recognition models in the API
process because that can trigger memory pressure and container restarts on
small cloud instances. We only need a yes/no gate for selfie upload here.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("identity_service")

_MAX_FACE_DIM = 1920
_FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


def _simple_embedding(face_bgr: np.ndarray) -> np.ndarray:
    """Create a small deterministic 512-d vector for compatibility."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (32, 16), interpolation=cv2.INTER_AREA)
    vec = tiny.astype(np.float32).reshape(-1)
    vec /= 255.0
    return vec


def extract_and_save(selfie_path: Path, embedding_path: Path) -> bool:
    """
    Detect the largest face in the selfie, save its embedding.
    Returns True if a face was found, False otherwise.
    """
    image = Image.open(selfie_path).convert("RGB")
    # Large phone photos can cause avoidable memory spikes during face analysis.
    if max(image.size) > _MAX_FACE_DIM:
        image.thumbnail((_MAX_FACE_DIM, _MAX_FACE_DIM), Image.Resampling.LANCZOS)
    bgr   = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if _FACE_CASCADE.empty():
        log.error("OpenCV face cascade failed to load")
        return False

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        log.warning(f"No face detected in {selfie_path}")
        return False

    # Pick the largest face by bounding-box area
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = bgr[y : y + h, x : x + w]
    embedding = _simple_embedding(crop)
    log.info(f"Face detected: bbox={[int(x), int(y), int(x + w), int(y + h)]}  embedding shape={embedding.shape}")

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    with open(embedding_path, "wb") as f:
        pickle.dump(embedding, f)

    return True


def load_embedding(embedding_path: Path) -> np.ndarray | None:
    if not embedding_path.exists():
        return None
    with open(embedding_path, "rb") as f:
        return pickle.load(f)
