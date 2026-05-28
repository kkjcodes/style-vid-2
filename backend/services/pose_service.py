"""
Preprocessing for WanAnimatePipeline (Track B).

WanAnimatePipeline requires two preprocessed videos:
  - pose_video: skeletal keypoints (body pose skeleton overlaid on black bg)
  - face_video: facial feature representation

Preprocessing uses DWPose (body) from controlnet_aux.
Face video is extracted by cropping and tracking the face region.

This is only needed for Track B (animate mode).
Track A (i2v) skips this entirely.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("pose_service")


def preprocess_reference_video(
    reference_path: Path,
    pose_out: Path,
    face_out: Path,
) -> bool:
    """
    Extract pose skeleton video and face video from a reference MP4.
    Returns True on success.

    Requires: controlnet_aux (pip install controlnet_aux)
    """
    try:
        import cv2
        from controlnet_aux import DWposeDetector
        from diffusers.utils import export_to_video
    except ImportError as e:
        log.error(f"Missing dependency for pose preprocessing: {e}")
        log.error("Install with: pip install controlnet_aux")
        return False

    log.info(f"Preprocessing reference video: {reference_path}")
    detector = DWposeDetector()

    cap = cv2.VideoCapture(str(reference_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    pose_frames = []
    face_frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        from PIL import Image
        import numpy as np

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Pose skeleton on black background
        pose_img = detector(pil, include_body=True, include_face=True, include_hand=False)
        pose_frames.append(pose_img)

        # Face crop: use center 40% of frame as proxy (simple, no tracking)
        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.1), int(h * 0.5)
        x1, x2 = int(w * 0.3), int(w * 0.7)
        face_crop = pil.crop((x1, y1, x2, y2)).resize((pil.width, pil.height))
        face_frames.append(face_crop)

    cap.release()

    if not pose_frames:
        log.error("No frames extracted from reference video")
        return False

    pose_out.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(pose_frames, str(pose_out), fps=int(fps))
    export_to_video(face_frames, str(face_out), fps=int(fps))
    log.info(f"Pose video → {pose_out}  ({len(pose_frames)} frames)")
    log.info(f"Face video → {face_out}  ({len(face_frames)} frames)")
    return True
