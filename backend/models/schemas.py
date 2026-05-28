from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class JobStatus(str, Enum):
    pending   = "pending"
    running   = "running"
    completed = "completed"
    failed    = "failed"


class OnboardResponse(BaseModel):
    user_id: str
    selfie_path: str
    face_detected: bool
    message: str


class GenerateI2VRequest(BaseModel):
    """Track A (simple): selfie → animated video via text prompt."""
    user_id: str
    prompt: str
    negative_prompt: str = (
        "blurry, low quality, static, worst quality, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, walking backwards"
    )
    num_frames: int = 81        # 4k+1; 81 = ~5s @ 16fps
    guidance_scale: float = 5.0


class GenerateAnimateRequest(BaseModel):
    """Track B (full): selfie + reference video → body-animated video."""
    user_id: str
    prompt: str
    negative_prompt: str = (
        "blurry, low quality, static, worst quality, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, walking backwards"
    )
    num_frames: int = 77        # WanAnimate uses 77 as default segment_frame_length
    guidance_scale: float = 1.0  # CFG disabled by default in WanAnimate


class GenerationResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: Optional[int] = None
    video_path: Optional[str] = None
    error: Optional[str] = None
