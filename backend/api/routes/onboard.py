from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, HttpUrl

from backend.models.schemas import OnboardResponse
from backend.services import storage_service, identity_service
from backend.core.logging_config import setup_logging

setup_logging()
log = logging.getLogger("routes.onboard")
router = APIRouter(prefix="/api/v1", tags=["onboard"])


@router.post("/onboard", response_model=OnboardResponse)
async def onboard(
    user_id: str = Form(...),
    selfie:  UploadFile = File(...),
):
    """
    Upload a selfie to register a user's identity.
    Validates that a face is detectable.
    """
    log.info(f"POST /onboard  user={user_id}  file={selfie.filename}")

    dest = storage_service.selfie_path(user_id)
    with open(dest, "wb") as f:
        shutil.copyfileobj(selfie.file, f)
    log.info(f"Selfie saved → {dest}")

    embed_path = dest.parent / "embedding.pkl"
    face_found = identity_service.extract_and_save(dest, embed_path)

    if not face_found:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="No face detected in the uploaded image. Please upload a clear front-facing photo."
        )

    return OnboardResponse(
        user_id=user_id,
        selfie_path=str(dest),
        face_detected=True,
        message="Selfie accepted. You can now generate videos.",
    )


@router.post("/onboard/reference")
async def upload_reference_video(
    user_id:   str = Form(...),
    reference: UploadFile = File(...),
):
    """
    Upload a reference video for Track B (body animation).
    The reference video provides the motion pattern to replicate.
    """
    log.info(f"POST /onboard/reference  user={user_id}  file={reference.filename}")

    dest = storage_service.reference_video_path(user_id)
    # Clear cached pose/face videos so they get re-extracted
    storage_service.pose_video_path(user_id).unlink(missing_ok=True)
    storage_service.face_video_path(user_id).unlink(missing_ok=True)

    with open(dest, "wb") as f:
        shutil.copyfileobj(reference.file, f)

    size_mb = dest.stat().st_size / 1e6
    log.info(f"Reference video saved → {dest} ({size_mb:.1f} MB)")
    return {"user_id": user_id, "reference_path": str(dest), "message": "Reference video saved."}


class ChannelRequest(BaseModel):
    user_id:    str
    youtube_url: str           # channel, playlist, or single video URL
    max_videos: int = 25
    num_steps:  int = 500
    lora_rank:  int = 16
    trigger_word: str = "sks"


@router.post("/onboard/channel")
async def start_style_training(body: ChannelRequest):
    """
    Track C: kick off LoRA training on the user's YouTube videos.
    Downloads up to max_videos clips and trains a style LoRA on them.
    Returns a job_id to poll with GET /api/v1/jobs/{job_id}.
    """
    from backend.workers.training_worker import run_training

    # Require selfie first — user must be onboarded before training
    selfie = storage_service.selfie_path(body.user_id)
    if not selfie.exists():
        raise HTTPException(
            status_code=422,
            detail="Upload a selfie via POST /api/v1/onboard before starting style training.",
        )

    log.info(f"POST /onboard/channel  user={body.user_id}  url={body.youtube_url}")

    task = run_training.delay(
        user_id=body.user_id,
        youtube_url=body.youtube_url,
        max_videos=body.max_videos,
        num_steps=body.num_steps,
        lora_rank=body.lora_rank,
        trigger_word=body.trigger_word,
    )

    return {
        "user_id":  body.user_id,
        "job_id":   task.id,
        "message":  (
            f"Style training queued. Downloading up to {body.max_videos} videos "
            f"then training for {body.num_steps} steps. "
            f"Poll GET /api/v1/jobs/{task.id} for status."
        ),
    }
