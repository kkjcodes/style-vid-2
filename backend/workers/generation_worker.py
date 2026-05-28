from __future__ import annotations

import logging
from pathlib import Path

from backend.workers.celery_app import celery_app
from backend.core.config import get_settings
from backend.core.logging_config import setup_logging

setup_logging()
log = logging.getLogger("generation_worker")
settings = get_settings()


@celery_app.task(bind=True, name="backend.workers.generation_worker.run_i2v")
def run_i2v(
    self,
    user_id: str,
    job_id: str,
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    guidance_scale: float,
):
    from backend.services import wan_service, storage_service

    log.info(f"[{job_id}] I2V start  user={user_id}")
    self.update_state(state="PROGRESS", meta={"progress_pct": 10})

    selfie  = storage_service.selfie_path(user_id)
    if not selfie.exists():
        raise FileNotFoundError(f"No selfie found for user {user_id}. Upload via /onboard first.")

    output  = storage_service.output_video_path(user_id, job_id)

    # Track C: use style LoRA if the user has one trained.
    # Auto-prepend trigger word so the user never has to think about it.
    lora = storage_service.lora_path(user_id)
    styled = lora.exists()
    if styled:
        trigger = storage_service.lora_trigger_word(user_id)
        if trigger and not prompt.startswith(trigger):
            prompt = f"{trigger} person, {prompt}"
            log.info(f"[{job_id}] Injected trigger word '{trigger}' → prompt='{prompt[:80]}'")

    self.update_state(state="PROGRESS", meta={"progress_pct": 20})
    wan_service.generate_i2v(
        selfie_path=selfie,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_path=output,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        lora_path=lora if styled else None,
    )

    size_mb = output.stat().st_size / 1e6
    log.info(f"[{job_id}] I2V done → {output} ({size_mb:.1f} MB)")
    return {"status": "completed", "video_path": str(output), "progress_pct": 100}


@celery_app.task(bind=True, name="backend.workers.generation_worker.run_animate")
def run_animate(
    self,
    user_id: str,
    job_id: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
):
    from backend.services import wan_service, storage_service, pose_service

    log.info(f"[{job_id}] Animate start  user={user_id}")
    self.update_state(state="PROGRESS", meta={"progress_pct": 5})

    selfie    = storage_service.selfie_path(user_id)
    reference = storage_service.reference_video_path(user_id)
    pose_vid  = storage_service.pose_video_path(user_id)
    face_vid  = storage_service.face_video_path(user_id)

    if not selfie.exists():
        raise FileNotFoundError(f"No selfie for user {user_id}")
    if not reference.exists():
        raise FileNotFoundError(f"No reference video for user {user_id}")

    if not pose_vid.exists() or not face_vid.exists():
        log.info(f"[{job_id}] Preprocessing reference video…")
        self.update_state(state="PROGRESS", meta={"progress_pct": 15})
        ok = pose_service.preprocess_reference_video(reference, pose_vid, face_vid)
        if not ok:
            raise RuntimeError("Pose preprocessing failed. Install controlnet_aux.")

    self.update_state(state="PROGRESS", meta={"progress_pct": 30})
    output = storage_service.output_video_path(user_id, job_id)

    wan_service.generate_animate(
        selfie_path=selfie,
        pose_video_path=pose_vid,
        face_video_path=face_vid,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_path=output,
        guidance_scale=guidance_scale,
    )

    size_mb = output.stat().st_size / 1e6
    log.info(f"[{job_id}] Animate done → {output} ({size_mb:.1f} MB)")
    return {"status": "completed", "video_path": str(output), "progress_pct": 100}
