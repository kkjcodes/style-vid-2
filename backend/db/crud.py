from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from backend.db.models import User, VideoJob
from backend.core.security import hash_password


def create_user(db: Session, username: str, password: str, email: str | None = None) -> User:
    user = User(
        id=str(uuid.uuid4()),
        username=username.lower().strip(),
        hashed_password=hash_password(password),
        email=email.lower().strip() if email else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username.lower().strip()).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email.lower().strip()).first()


def get_user_by_id(db: Session, user_id: str) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def set_reset_token(db: Session, user_id: str, token: str, expires: "datetime") -> None:
    db.query(User).filter(User.id == user_id).update({
        "reset_token": token,
        "reset_token_expires": expires,
    })
    db.commit()


def clear_reset_token(db: Session, user_id: str) -> None:
    db.query(User).filter(User.id == user_id).update({
        "reset_token": None,
        "reset_token_expires": None,
    })
    db.commit()


def update_password(db: Session, user_id: str, new_hashed: str) -> None:
    db.query(User).filter(User.id == user_id).update({"hashed_password": new_hashed})
    db.commit()


def set_replicate_key(db: Session, user_id: str, encrypted_key: str) -> None:
    db.query(User).filter(User.id == user_id).update(
        {"replicate_key_encrypted": encrypted_key}
    )
    db.commit()


def set_replicate_username(db: Session, user_id: str, username: str) -> None:
    db.query(User).filter(User.id == user_id).update({"replicate_username": username})
    db.commit()


def set_frames_training_job(db: Session, user_id: str, training_id: str) -> None:
    db.query(User).filter(User.id == user_id).update({"frames_training_id": training_id})
    db.commit()


def set_lora_celery_job(db: Session, user_id: str, job_id: str) -> None:
    db.query(User).filter(User.id == user_id).update({
        "lora_celery_job_id": job_id,
        "lora_status": "pending",
    })
    db.commit()


def set_lora_training_started(
    db: Session,
    user_id: str,
    training_id: str,
    model_ref: str,
    trigger_word: str,
) -> None:
    db.query(User).filter(User.id == user_id).update({
        "lora_training_id":  training_id,
        "lora_model_ref":    model_ref,
        "lora_trigger_word": trigger_word,
        "lora_status":       "running",
        "lora_version_id":   None,
        "lora_error":        None,
    })
    db.commit()


def set_lora_training_complete(
    db: Session, user_id: str, version_id: str, weights_url: str | None = None
) -> None:
    db.query(User).filter(User.id == user_id).update({
        "lora_version_id":  version_id,
        "lora_weights_url": weights_url,
        "lora_status":      "completed",
        "lora_error":       None,
    })
    db.commit()


def set_lora_training_failed(db: Session, user_id: str, error: str) -> None:
    db.query(User).filter(User.id == user_id).update({
        "lora_status": "failed",
        "lora_error":  error[:500],
    })
    db.commit()


def clear_lora(db: Session, user_id: str) -> None:
    db.query(User).filter(User.id == user_id).update({
        "lora_model_ref":    None,
        "lora_version_id":   None,
        "lora_trigger_word": None,
        "lora_training_id":  None,
        "lora_status":       None,
        "lora_error":        None,
    })
    db.commit()


def delete_user(db: Session, user_id: str) -> None:
    db.query(VideoJob).filter(VideoJob.user_id == user_id).delete()
    db.query(User).filter(User.id == user_id).delete()
    db.commit()


# ── VideoJob ──────────────────────────────────────────────────────────────────

def create_video_job(
    db: Session,
    job_id: str,
    user_id: str,
    prompt: str,
    resolution: str,
    num_clips: int,
) -> VideoJob:
    job = VideoJob(
        id=job_id,
        user_id=user_id,
        prompt=prompt,
        resolution=resolution,
        num_clips=num_clips,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job_by_id(db: Session, job_id: str) -> VideoJob | None:
    return db.query(VideoJob).filter(VideoJob.id == job_id).first()


def delete_video_job(db: Session, job_id: str, user_id: str) -> int:
    """Delete one video job owned by the given user. Returns number of deleted rows."""
    deleted = (
        db.query(VideoJob)
        .filter(VideoJob.id == job_id, VideoJob.user_id == user_id)
        .delete()
    )
    db.commit()
    return deleted


def get_user_video_jobs(db: Session, user_id: str) -> list[VideoJob]:
    return (
        db.query(VideoJob)
        .filter(VideoJob.user_id == user_id)
        .order_by(VideoJob.created_at.desc())
        .all()
    )


def sync_video_job(
    db: Session,
    job_id: str,
    status: str,
    progress_pct: int = 0,
    video_filename: str | None = None,
    error_message: str | None = None,
) -> None:
    updates: dict = {"status": status, "progress_pct": progress_pct}
    if video_filename:
        updates["video_filename"] = video_filename
        updates["completed_at"] = datetime.utcnow()
    if error_message:
        updates["error_message"] = error_message
    db.query(VideoJob).filter(VideoJob.id == job_id).update(updates)
    db.commit()
