"""
Training API routes.

Frame extraction (fast, free — for r2v mode):
  POST   /api/v1/training/start
  GET    /api/v1/training/status/{job_id}
  GET    /api/v1/training/frames
  DELETE /api/v1/training/frames

LoRA fine-tuning (best identity, ~$5-10 one-time per user):
  POST   /api/v1/training/lora/start
  GET    /api/v1/training/lora/status/{job_id}
  GET    /api/v1/training/lora
  DELETE /api/v1/training/lora
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.core.logging_config import setup_logging
from backend.db import crud
from backend.db.database import get_db
from backend.db.models import User
from backend.services import storage_service

setup_logging()
log = logging.getLogger("routes.training")
router = APIRouter(prefix="/api/v1/training", tags=["training"])


_ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_ALLOWED_VIDEO_MIME = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/x-matroska", "video/webm",
}
_MAX_VIDEO_BYTES = 100 * 1024 * 1024   # 100 MB
_MAX_VIDEO_FILES = 5


@router.post("/upload-videos")
async def upload_training_videos(
    videos: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload 1–5 MP4/MOV video files for face training.
    Replaces any previously uploaded videos.
    Files are deleted from the server once training is submitted to Replicate.
    """
    if not videos:
        raise HTTPException(422, "Upload at least one video file.")
    if len(videos) > _MAX_VIDEO_FILES:
        raise HTTPException(422, f"Maximum {_MAX_VIDEO_FILES} videos per upload.")

    video_dir = storage_service.training_video_dir(current_user.id)
    # Clear existing uploads before saving new ones
    for old in video_dir.iterdir():
        if old.is_file():
            old.unlink(missing_ok=True)

    saved = []
    for v in videos:
        ext = Path(v.filename or "").suffix.lower()
        if ext not in _ALLOWED_VIDEO_EXTS and v.content_type not in _ALLOWED_VIDEO_MIME:
            raise HTTPException(
                422,
                f"Unsupported format '{v.filename}'. Use MP4 or MOV.",
            )
        safe_name = f"{uuid.uuid4().hex}{ext or '.mp4'}"
        dest = video_dir / safe_name
        size = 0
        with open(dest, "wb") as f:
            while chunk := await v.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_VIDEO_BYTES:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"'{v.filename}' exceeds the 100 MB limit.",
                    )
                f.write(chunk)
        saved.append({"original_name": v.filename, "size_mb": round(size / 1e6, 1)})
        log.info(
            f"Uploaded training video: user={current_user.username} "
            f"file={v.filename} size={size // 1024}KB"
        )

    return {
        "uploaded": len(saved),
        "files": saved,
        "message": (
            f"{len(saved)} video(s) ready. "
            "Files are deleted once training is submitted to Replicate."
        ),
    }


@router.get("/videos")
def list_training_videos(current_user: User = Depends(get_current_user)):
    """Check whether this user has training videos uploaded on the server."""
    videos = storage_service.training_video_paths(current_user.id)
    total_mb = sum(v.stat().st_size for v in videos) / 1e6 if videos else 0.0
    return {
        "has_videos": len(videos) > 0,
        "count": len(videos),
        "total_mb": round(total_mb, 1),
    }


@router.delete("/videos")
def delete_training_videos(current_user: User = Depends(get_current_user)):
    """Delete all uploaded training videos for this user."""
    storage_service.delete_training_videos(current_user.id)
    return {"message": "Training videos cleared."}


class StartTrainingRequest(BaseModel):
    youtube_urls: list[str] = []   # kept for backward compat; prefer file upload
    max_frames: int = 20


@router.post("/start")
def start_training(
    body: StartTrainingRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Kick off reference frame extraction from uploaded videos."""
    from backend.workers.training_worker import extract_reference_frames

    has_local = bool(storage_service.training_video_paths(current_user.id))
    if not body.youtube_urls and not has_local:
        raise HTTPException(422, "Upload videos first.")

    job_id = str(uuid.uuid4())
    crud.set_frames_training_job(db, current_user.id, job_id)
    log.info(
        f"POST /training/start  user={current_user.username}  "
        f"urls={len(body.youtube_urls)}  job={job_id}"
    )

    extract_reference_frames.apply_async(
        kwargs={
            "user_id":      current_user.id,
            "youtube_urls": body.youtube_urls,
            "max_frames":   body.max_frames,
        },
        queue="generation",
        task_id=job_id,
    )

    return {
        "job_id":  job_id,
        "status":  "pending",
        "message": "Extracting reference frames. Takes 2–5 min. Poll /training/status/{job_id}",
    }


@router.get("/status/{job_id}")
def training_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Poll the reference frame extraction job."""
    from backend.workers.celery_app import celery_app

    if current_user.frames_training_id != job_id:
        raise HTTPException(403, "Not your training job.")

    result = celery_app.AsyncResult(job_id)
    info = result.info if isinstance(result.info, dict) else {}

    state_map = {
        "PENDING":  "pending",
        "STARTED":  "running",
        "PROGRESS": "running",
        "SUCCESS":  "completed",
        "FAILURE":  "failed",
    }

    return {
        "job_id":      job_id,
        "status":      state_map.get(result.state, "pending"),
        "progress_pct": info.get("progress_pct", 0),
        "stage":       info.get("stage", "queued"),
        "frame_count": info.get("frame_count") if result.state == "SUCCESS" else None,
        "error":       str(result.info) if result.state == "FAILURE" else None,
    }


@router.get("/frames")
def list_reference_frames(current_user: User = Depends(get_current_user)):
    """Return metadata for the current user's extracted reference frames."""
    frames = storage_service.reference_frame_paths(current_user.id)
    return {
        "frame_count": len(frames),
        "has_frames":  len(frames) > 0,
        "frames": [{"filename": p.name} for p in frames],
    }


@router.delete("/frames")
def delete_reference_frames(current_user: User = Depends(get_current_user)):
    """Delete reference frames so new YouTube videos can be used."""
    frames_dir = Path(storage_service.reference_frames_dir(current_user.id))
    deleted = 0
    for f in frames_dir.glob("frame_*.jpg"):
        f.unlink(missing_ok=True)
        deleted += 1
    log.info(f"Cleared {deleted} reference frames for user={current_user.username}")
    return {"deleted": deleted, "message": "Reference frames cleared. Upload new videos to re-extract."}


# ─── LoRA training ────────────────────────────────────────────────────────────

class LoraStartRequest(BaseModel):
    youtube_urls: list[str] = []   # kept for backward compat; prefer file upload


@router.post("/lora/start")
def start_lora_training(
    body: LoraStartRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Kick off LoRA fine-tuning from uploaded videos.
    ~10–15 min on Replicate H100, costs ~$5–10 from your Replicate balance.
    """
    from backend.workers.training_worker import run_lora_training

    has_local = bool(storage_service.training_video_paths(current_user.id))
    if not body.youtube_urls and not has_local:
        raise HTTPException(422, "Upload videos first using the 'Boost Quality' section.")

    if not current_user.replicate_key_encrypted:
        raise HTTPException(400, "Save your Replicate API key first (Account page).")

    selfie = storage_service.selfie_path(current_user.id)
    if not selfie.exists():
        raise HTTPException(400, "Upload a selfie first (Setup page).")

    if current_user.lora_status == "running" and current_user.lora_training_id:
        raise HTTPException(409, "A LoRA training run is already in progress for this account.")

    job_id = str(uuid.uuid4())
    log.info(f"POST /training/lora/start  user={current_user.username}  job={job_id}")

    crud.set_lora_celery_job(db, current_user.id, job_id)

    run_lora_training.apply_async(
        kwargs={
            "user_id":      current_user.id,
            "youtube_urls": body.youtube_urls,
        },
        queue="generation",
        task_id=job_id,
    )

    return {
        "job_id":  job_id,
        "status":  "pending",
        "message": (
            "LoRA training queued. This takes ~10–15 min and costs ~$5–10 from "
            "your Replicate balance. Poll /training/lora/status/{job_id} for progress."
        ),
    }


@router.get("/lora/status/{job_id}")
def lora_training_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Poll LoRA training progress.

    Phase 1 (prep — download/clip/zip/submit): reads Celery task state.
    Phase 2 (Replicate training): reads DB state updated by check_lora_training poller.
    """
    from backend.workers.celery_app import celery_app
    from backend.db import crud as _crud

    if current_user.lora_celery_job_id != job_id:
        raise HTTPException(403, "Not your LoRA training job.")

    result = celery_app.AsyncResult(job_id)
    info   = result.info if isinstance(result.info, dict) else {}

    # Prep phase: task still in flight
    if result.state in ("PENDING", "STARTED", "PROGRESS"):
        prep_status = "pending" if result.state == "PENDING" else "running"
        return {
            "job_id":       job_id,
            "status":       prep_status,
            "progress_pct": info.get("progress_pct", 0),
            "stage":        info.get("stage", "queued"),
            "model_ref":    None,
            "error":        None,
        }

    # Prep phase failed before submitting to Replicate
    if result.state == "FAILURE":
        return {
            "job_id":       job_id,
            "status":       "failed",
            "progress_pct": 0,
            "stage":        "failed",
            "model_ref":    None,
            "error":        str(result.info),
        }

    # Prep task done (SUCCESS/submitted) — read Replicate status from DB
    fresh = _crud.get_user_by_id(db, current_user.id)
    db_status   = (fresh.lora_status or "running") if fresh else "running"
    db_error    = fresh.lora_error if fresh else None
    db_ref      = fresh.lora_model_ref if fresh else None
    db_weights  = fresh.lora_weights_url if fresh else None
    pct_map     = {"pending": 55, "running": 70, "completed": 100, "failed": 0}
    return {
        "job_id":       job_id,
        "status":       db_status,
        "progress_pct": pct_map.get(db_status, 55),
        "stage":        "Training on Replicate…" if db_status == "running" else db_status,
        "model_ref":    db_ref if db_status == "completed" else None,
        "weights_url":  db_weights if db_status == "completed" else None,
        "error":        db_error,
    }


@router.get("/lora")
def get_lora_status(
    current_user: User = Depends(get_current_user),
):
    """Return the current LoRA model status for this user."""
    return {
        "job_id":       current_user.lora_celery_job_id,
        "status":       current_user.lora_status,
        "model_ref":    current_user.lora_model_ref,
        "version_id":   current_user.lora_version_id,
        "trigger_word": current_user.lora_trigger_word,
        "error":        current_user.lora_error,
        "trained":      current_user.lora_status == "completed" and bool(current_user.lora_version_id),
    }


@router.delete("/lora")
def delete_lora(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Reset LoRA training state so a fresh training run can be started."""
    crud.clear_lora(db, current_user.id)
    log.info(f"LoRA cleared for user={current_user.username}")
    return {"message": "LoRA training state cleared. You can start a new training run."}
