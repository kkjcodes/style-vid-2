"""
Frame-level face swap using InsightFace inswapper.

For each frame in the generated video, any detected face is replaced with
the user's selfie face. This locks identity that drifts during video diffusion.

Model auto-downloaded on first use (~270 MB):
  ~/.insightface/models/inswapper_128.onnx
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("face_swap_service")

_app = None
_swapper = None


def _load_models():
    global _app, _swapper
    if _app is not None:
        return _app, _swapper

    import insightface
    from insightface.app import FaceAnalysis

    log.info("Loading InsightFace models (first run downloads ~270 MB)…")
    _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=0, det_size=(640, 640))

    _swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx", download=True, download_zip=False
    )
    log.info("InsightFace models ready.")
    return _app, _swapper


def _get_best_face(faces):
    """Return the face with the largest bounding box (most prominent face)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def swap_video(
    video_path: Path,
    source_face_path: Path,
    output_path: Path,
    skip_frames: int = 0,
) -> Path:
    """
    Replace every detected face in video_path with the face from source_face_path.

    skip_frames: process every (skip_frames+1)th frame; interpolate the rest.
                 0 = process every frame (best quality, slowest).
                 2 = process every 3rd frame (3x faster, minimal quality loss).
    Returns output_path.
    """
    app, swapper = _load_models()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Get source face embedding ──────────────────────────────────────────────
    source_img = cv2.imread(str(source_face_path))
    if source_img is None:
        raise ValueError(f"Cannot read source face image: {source_face_path}")

    source_faces = app.get(source_img)
    if not source_faces:
        raise ValueError("No face detected in source selfie. Upload a clear front-facing photo.")
    source_face = _get_best_face(source_faces)
    log.info(f"Source face detected  det_score={source_face.det_score:.2f}")

    # ── Open video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 16.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write frames to a temp AVI with XVID — mp4v on macOS produces green frames.
    # ffmpeg re-encodes to H.264 mp4 afterward.
    tmp_avi = Path(tempfile.mktemp(suffix=".avi", dir=output_path.parent))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(tmp_avi), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open {tmp_avi} — check OpenCV codec support")

    log.info(f"Face swap: {total} frames  {width}×{height}  fps={fps:.1f}")

    frame_idx = 0
    swapped_count = 0
    skipped_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        should_swap = (skip_frames == 0) or (frame_idx % (skip_frames + 1) == 0)

        if should_swap:
            target_faces = app.get(frame)
            if target_faces:
                target_face = _get_best_face(target_faces)
                frame = swapper.get(frame, target_face, source_face, paste_back=True)
                swapped_count += 1
        else:
            skipped_count += 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    if not tmp_avi.exists() or tmp_avi.stat().st_size == 0:
        raise RuntimeError(f"Face swap produced empty intermediate file: {tmp_avi}")

    # Re-encode AVI → H.264 mp4 (avoids mp4v green-screen on macOS)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(tmp_avi),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ],
        capture_output=True, text=True,
    )
    tmp_avi.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed:\n{result.stderr[-500:]}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty output: {output_path}")

    log.info(
        f"Face swap done: {swapped_count}/{frame_idx} frames swapped  "
        f"({skipped_count} skipped)  → {output_path}"
    )
    return output_path
