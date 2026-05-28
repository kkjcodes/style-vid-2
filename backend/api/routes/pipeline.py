"""
Pipeline API routes — Replicate-based video generation.

POST /api/v1/pipeline/selfie              upload selfie (JWT required)
POST /api/v1/pipeline/generate            kick off video generation (JWT required)
GET  /api/v1/pipeline/jobs/{job_id}       poll job status
GET  /api/v1/pipeline/history             list current user's past jobs (JWT required)
DELETE /api/v1/pipeline/history/{job_id}  delete one generated video + DB record (JWT required)
GET  /api/v1/pipeline/video/{filename}    stream/download a generated video (JWT required)
POST /api/v1/pipeline/test-key            validate a Replicate key (no auth needed)
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.core.config import get_settings
from backend.core.logging_config import setup_logging
from backend.db import crud
from backend.db.database import get_db
from backend.db.models import User
from backend.services import storage_service, identity_service

setup_logging()
log = logging.getLogger("routes.pipeline")
settings = get_settings()
router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


class TestKeyRequest(BaseModel):
    replicate_key: str


class GenerateRequest(BaseModel):
    prompt: str
    num_clips: int = 2
    resolution: str = "720p"

    @field_validator("num_clips")
    @classmethod
    def _validate_num_clips(cls, value: int) -> int:
        if value < 1 or value > 6:
            raise ValueError("num_clips must be between 1 and 6")
        return value

    @field_validator("resolution")
    @classmethod
    def _validate_resolution(cls, value: str) -> str:
        if value not in {"480p", "720p", "1080p"}:
            raise ValueError("resolution must be one of 480p, 720p, 1080p")
        return value


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _video_url(filename: str | None) -> str | None:
    """Return a usable video URL from a stored filename or blob URL."""
    if not filename:
        return None
    if filename.startswith("https://"):
        return filename  # already a blob URL
    return f"/api/v1/pipeline/video/{filename}"


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/test-key")
def test_replicate_key(body: TestKeyRequest):
    """Validate a Replicate API key without generating anything."""
    from backend.services.replicate_service import test_connection
    valid = test_connection(body.replicate_key)
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid Replicate API key.")
    return {"valid": True, "message": "Replicate key verified ✓"}


_ALLOWED_SELFIE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
_ALLOWED_SELFIE_MIME = {
    "image/jpeg", "image/png", "image/webp",
    "image/heic", "image/heif",
}


@router.post("/selfie")
async def upload_selfie(
    selfie: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload selfie and validate that a face is detected."""
    ext = Path(selfie.filename or "").suffix.lower()
    if ext not in _ALLOWED_SELFIE_EXTS and selfie.content_type not in _ALLOWED_SELFIE_MIME:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext or selfie.content_type}'. "
                   "Please upload a JPEG, PNG, WebP, or HEIC photo.",
        )

    dest = storage_service.selfie_path(current_user.id)
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(selfie.file, f)
    except Exception as exc:
        log.warning(f"Selfie save failed for user={current_user.username}: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Temporary storage issue while saving selfie. Please try again.",
        )
    finally:
        await selfie.close()

    embed_path = dest.parent / "embedding.pkl"
    try:
        face_found = identity_service.extract_and_save(dest, embed_path)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        log.warning(f"Selfie processing failed for user={current_user.username}: {exc}")
        raise HTTPException(
            status_code=422,
            detail="Could not read image. Please upload a JPEG or PNG — HEIC may need to be converted first.",
        )

    if not face_found:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="No face detected. Upload a clear, well-lit front-facing photo.",
        )

    log.info(f"Selfie uploaded: user={current_user.username}")
    return {"face_detected": True, "message": "Selfie accepted ✓"}


@router.post("/generate")
def start_generation(
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Kick off async video generation. Returns job_id for polling."""
    from backend.workers.pipeline_worker import run_pipeline

    if not current_user.replicate_key_encrypted:
        raise HTTPException(
            status_code=422,
            detail="Add your Replicate API key in Account settings first.",
        )

    selfie = storage_service.selfie_path(current_user.id)
    if not selfie.exists():
        raise HTTPException(status_code=422, detail="Upload a selfie first in Setup.")

    job_id = str(uuid.uuid4())

    crud.create_video_job(
        db, job_id, current_user.id, body.prompt, body.resolution, body.num_clips
    )

    log.info(
        f"POST /pipeline/generate  user={current_user.username}  "
        f"job={job_id}  clips={body.num_clips}  res={body.resolution}"
    )
    run_pipeline.apply_async(
        kwargs={
            "user_id":       current_user.id,
            "job_id":        job_id,
            "prompt":        body.prompt,
            "num_clips":     body.num_clips,
            "resolution":    body.resolution,
        },
        queue="generation",
        task_id=job_id,
    )

    return {
        "job_id":           job_id,
        "status":           "pending",
        "estimated_minutes": body.num_clips * 2,
        "video_duration_sec": body.num_clips * 5,
        "message": f"Generating {body.num_clips * 5}s {body.resolution} video.",
    }


@router.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Poll generation job status. Also syncs Celery state → DB."""
    from backend.workers.celery_app import celery_app

    # Ownership check — prevent polling another user's job
    existing = crud.get_job_by_id(db, job_id)
    if existing and existing.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your job.")

    result = celery_app.AsyncResult(job_id)
    info = result.info if isinstance(result.info, dict) else {}

    state_map = {
        "PENDING":  "pending",
        "STARTED":  "running",
        "PROGRESS": "running",
        "SUCCESS":  "completed",
        "FAILURE":  "failed",
    }
    status = state_map.get(result.state, "pending")

    # Sync terminal states back to DB
    video_filename = None
    if result.state == "SUCCESS":
        video_path = info.get("video_path")
        # video_path is either a local filesystem path or a blob URL
        if video_path and video_path.startswith("https://"):
            video_filename = video_path
        else:
            video_filename = Path(video_path).name if video_path else None
        crud.sync_video_job(
            db, job_id, "completed",
            progress_pct=100,
            video_filename=video_filename,
        )
    elif result.state == "FAILURE":
        crud.sync_video_job(
            db, job_id, "failed",
            error_message=str(result.info),
        )
    elif result.state == "PROGRESS":
        crud.sync_video_job(db, job_id, "running", progress_pct=info.get("progress_pct", 0))

    return {
        "job_id":       job_id,
        "status":       status,
        "progress_pct": info.get("progress_pct", 0),
        "message":      info.get("message", ""),
        "video_url":    _video_url(video_filename) if result.state == "SUCCESS" else None,
        "error":        str(result.info) if result.state == "FAILURE" else None,
    }


@router.get("/history")
def list_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all past video generation jobs for the current user."""
    jobs = crud.get_user_video_jobs(db, current_user.id)
    return {
        "jobs": [
            {
                "job_id":       j.id,
                "prompt":       j.prompt,
                "status":       j.status,
                "resolution":   j.resolution,
                "num_clips":    j.num_clips,
                "progress_pct": j.progress_pct,
                "video_url":    _video_url(j.video_filename),
                "created_at":   j.created_at.isoformat() if j.created_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "error":        j.error_message,
            }
            for j in jobs
        ]
    }


@router.delete("/history/{job_id}")
def delete_history_item(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete one generated video from storage and remove its DB job row."""
    job = crud.get_job_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your job.")

    storage_deleted = storage_service.delete_generated_video(current_user.id, job.video_filename)
    db_deleted = crud.delete_video_job(db, job_id, current_user.id)
    if db_deleted == 0:
        raise HTTPException(status_code=404, detail="Job not found.")

    log.info(
        f"Deleted history item: user={current_user.username} job={job_id} "
        f"storage_deleted={storage_deleted}"
    )
    return {
        "deleted": True,
        "job_id": job_id,
        "storage_deleted": storage_deleted,
    }


@router.get("/video/{filename}")
def serve_video(
    filename: str,
    current_user: User = Depends(get_current_user),
):
    """Stream/download a generated video file (dev only — prod uses blob URLs directly)."""
    # Strip all directory components to prevent path traversal
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    # Reject anything that looks like traversal even after name extraction
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    video_path = (
        Path(settings.local_storage_dir)
        / "users" / current_user.id / "outputs"
        / safe_name
    )
    # Resolve and confirm the path stays inside the user's output directory
    user_output_dir = (Path(settings.local_storage_dir) / "users" / current_user.id / "outputs").resolve()
    try:
        resolved = video_path.resolve()
        resolved.relative_to(user_output_dir)  # raises ValueError if outside
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")

    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(str(resolved), media_type="video/mp4", filename=safe_name)
