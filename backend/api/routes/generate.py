from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    GenerateI2VRequest,
    GenerateAnimateRequest,
    GenerationResponse,
    JobStatus,
    JobStatusResponse,
)
from backend.workers.celery_app import celery_app
from backend.workers.generation_worker import run_i2v, run_animate
from backend.core.logging_config import setup_logging

setup_logging()
log = logging.getLogger("routes.generate")
router = APIRouter(prefix="/api/v1", tags=["generate"])


@router.post("/generate/i2v", response_model=GenerationResponse)
def start_i2v(body: GenerateI2VRequest):
    """Track A: selfie + text prompt → video."""
    job_id = str(uuid.uuid4())
    log.info(f"POST /generate/i2v  user={body.user_id}  job={job_id}")
    run_i2v.apply_async(
        kwargs={
            "user_id":         body.user_id,
            "job_id":          job_id,
            "prompt":          body.prompt,
            "negative_prompt": body.negative_prompt,
            "num_frames":      body.num_frames,
            "guidance_scale":  body.guidance_scale,
        },
        queue="generation",
        task_id=job_id,
    )
    return GenerationResponse(job_id=job_id, status=JobStatus.pending)


@router.post("/generate/animate", response_model=GenerationResponse)
def start_animate(body: GenerateAnimateRequest):
    """Track B: selfie + reference video → body-animated video."""
    job_id = str(uuid.uuid4())
    log.info(f"POST /generate/animate  user={body.user_id}  job={job_id}")
    run_animate.apply_async(
        kwargs={
            "user_id":         body.user_id,
            "job_id":          job_id,
            "prompt":          body.prompt,
            "negative_prompt": body.negative_prompt,
            "guidance_scale":  body.guidance_scale,
        },
        queue="generation",
        task_id=job_id,
    )
    return GenerationResponse(job_id=job_id, status=JobStatus.pending)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    try:
        result = celery_app.AsyncResult(job_id)
        state_map = {
            "PENDING":  JobStatus.pending,
            "STARTED":  JobStatus.running,
            "PROGRESS": JobStatus.running,
            "SUCCESS":  JobStatus.completed,
            "FAILURE":  JobStatus.failed,
        }
        status = state_map.get(result.state, JobStatus.pending)
        info   = result.info if isinstance(result.info, dict) else {}

        return JobStatusResponse(
            job_id=job_id,
            status=status,
            progress_pct=info.get("progress_pct"),
            video_path=info.get("video_path"),
            error=str(result.info) if result.state == "FAILURE" else None,
        )
    except Exception as e:
        log.error(f"GET /jobs/{job_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
