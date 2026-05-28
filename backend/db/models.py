from __future__ import annotations
from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String, UniqueConstraint
from backend.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    replicate_key_encrypted = Column(String, nullable=True)
    email = Column(String, nullable=True, index=True)           # required for new accounts
    reset_token = Column(String, nullable=True)                 # hex token, single-use
    reset_token_expires = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Replicate account (cached after first lookup)
    replicate_username = Column(String, nullable=True)

    # LoRA training state
    frames_training_id = Column(String, nullable=True)   # Celery job id for reference-frame extraction
    lora_model_ref    = Column(String, nullable=True)   # e.g. "alice/stylevid-a1b2"
    lora_version_id   = Column(String, nullable=True)   # specific version for inference
    lora_trigger_word = Column(String, nullable=True)   # e.g. "SBJTabc123"
    lora_celery_job_id = Column(String, nullable=True)   # Celery task UUID (returned to client)
    lora_training_id  = Column(String, nullable=True)   # Replicate training ID (set by worker)
    lora_weights_url  = Column(String, nullable=True)   # direct URL to .safetensors (Option B inference)
    lora_status       = Column(String, nullable=True)   # pending|running|completed|failed
    lora_error        = Column(String, nullable=True)


class VideoJob(Base):
    __tablename__ = "video_jobs"

    id = Column(String, primary_key=True)          # UUID — doubles as Celery task_id
    user_id = Column(String, nullable=False, index=True)
    prompt = Column(String, nullable=False)
    status = Column(String, default="pending")      # pending|running|completed|failed
    progress_pct = Column(Integer, default=0)
    resolution = Column(String, default="720p")
    num_clips = Column(Integer, default=2)
    video_filename = Column(String, nullable=True)  # set on completion
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
