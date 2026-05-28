"""
Celery task: full Replicate video generation pipeline.

Triggered by POST /api/v1/pipeline/generate.
Stores progress in Celery task state so the frontend can poll it.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.workers.celery_app import celery_app
from backend.core.config import get_settings
from backend.core.logging_config import setup_logging
from backend.core.security import decrypt_key
from backend.services import storage_service

setup_logging()
log = logging.getLogger("pipeline_worker")
settings = get_settings()


@celery_app.task(
    bind=True,
    name="backend.workers.pipeline_worker.run_pipeline",
    soft_time_limit=3600,  # 60 min cap
    time_limit=3660,
)
def run_pipeline(
    self,
    user_id: str,
    job_id: str,
    prompt: str,
    num_clips: int = 3,
    resolution: str = "480p",
):
    """
    Full pipeline:
      1. Load selfie from storage
      2. For each clip: generate via Replicate + face swap
      3. Stitch clips → final MP4
      4. Save to storage and return path
    """
    from backend.services import replicate_service
    from backend.db.database import SessionLocal
    from backend.db import crud

    selfie = storage_service.selfie_path(user_id)
    if not selfie.exists():
        raise FileNotFoundError(f"No selfie for user {user_id}. Upload one first.")

    output = storage_service.output_video_path(user_id, job_id)

    def _progress(pct: int, message: str):
        log.info(f"[{job_id}] {pct}%  {message}")
        self.update_state(
            state="PROGRESS",
            meta={"progress_pct": pct, "message": message},
        )

    # Determine generation mode: LoRA > r2v > i2v
    db = SessionLocal()
    try:
        user = crud.get_user_by_id(db, user_id)
        lora_model_ref    = user.lora_model_ref    if user else None
        lora_version_id   = user.lora_version_id   if user else None
        lora_trigger_word = user.lora_trigger_word  if user else None
        lora_weights_url  = user.lora_weights_url   if user else None
        if not user or not user.replicate_key_encrypted:
            raise RuntimeError("Replicate key missing for this user.")
        replicate_key = decrypt_key(user.replicate_key_encrypted)
    finally:
        db.close()

    use_lora = bool(lora_model_ref and lora_version_id)
    ref_frames = [] if use_lora else storage_service.reference_frame_paths(user_id)

    if use_lora:
        mode = "lora (trained)"
    elif ref_frames:
        mode = "reference (r2v)"
    else:
        mode = "selfie (i2v)"

    _progress(5, f"Starting pipeline… mode={mode}")
    log.info(f"[{job_id}] mode={mode}  ref_frames={len(ref_frames)}")

    replicate_service.generate_full_video(
        api_key=replicate_key,
        selfie_path=selfie,
        prompt=prompt,
        output_path=output,
        reference_frame_paths=ref_frames if ref_frames else None,
        lora_model_ref=lora_model_ref,
        lora_version_id=lora_version_id,
        lora_trigger_word=lora_trigger_word,
        lora_weights_url=lora_weights_url,
        num_clips=num_clips,
        resolution=resolution,
        on_progress=_progress,
    )

    size_mb = output.stat().st_size / 1e6
    log.info(f"[{job_id}] Pipeline done → {output}  ({size_mb:.1f} MB)")

    from backend.services import blob_service
    if blob_service.is_enabled():
        video_ref = blob_service.upload_video(user_id, output.name, output)
    else:
        video_ref = str(output)

    return {
        "status": "completed",
        "video_path": video_ref,
        "progress_pct": 100,
        "message": "Video ready!",
    }
