from __future__ import annotations
import logging
from pathlib import Path
from backend.core.config import get_settings

log = logging.getLogger("storage_service")
settings = get_settings()


def _root() -> Path:
    p = Path(settings.local_storage_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def selfie_path(user_id: str) -> Path:
    p = _root() / "users" / user_id / "selfie"
    p.mkdir(parents=True, exist_ok=True)
    return p / "selfie.jpg"


def reference_video_path(user_id: str) -> Path:
    p = _root() / "users" / user_id / "reference"
    p.mkdir(parents=True, exist_ok=True)
    return p / "reference.mp4"


def pose_video_path(user_id: str) -> Path:
    return _root() / "users" / user_id / "reference" / "pose.mp4"


def face_video_path(user_id: str) -> Path:
    return _root() / "users" / user_id / "reference" / "face.mp4"


def output_video_path(user_id: str, job_id: str) -> Path:
    p = _root() / "users" / user_id / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{job_id}.mp4"


def lora_path(user_id: str) -> Path:
    """Return the path to the user's trained LoRA weights if they exist."""
    lora_dir = _root() / "loras" / user_id
    # Prefer safetensors; fall back to .pt
    st = lora_dir / "lora_weights.safetensors"
    pt = lora_dir / "lora_weights.pt"
    return st if st.exists() else pt


def lora_output_dir(user_id: str) -> Path:
    p = _root() / "loras" / user_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def lora_trigger_word(user_id: str) -> str | None:
    """Return the trigger word saved during training, or None if not trained yet."""
    f = _root() / "loras" / user_id / "trigger_word.txt"
    return f.read_text().strip() if f.exists() else None


def reference_frames_dir(user_id: str) -> Path:
    p = _root() / "users" / user_id / "reference_frames"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reference_frame_paths(user_id: str) -> list[Path]:
    """Return sorted list of extracted reference frame paths for this user."""
    d = reference_frames_dir(user_id)
    return sorted(d.glob("frame_*.jpg"))


def has_reference_frames(user_id: str) -> bool:
    return len(reference_frame_paths(user_id)) > 0


def training_zip_path(user_id: str) -> Path:
    """Temporary zip of face clips sent to Replicate for LoRA training."""
    p = _root() / "users" / user_id / "lora_prep"
    p.mkdir(parents=True, exist_ok=True)
    return p / "training.zip"


def training_video_dir(user_id: str) -> Path:
    p = _root() / "users" / user_id / "training_videos"
    p.mkdir(parents=True, exist_ok=True)
    return p


_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def training_video_paths(user_id: str) -> list[Path]:
    """Return all uploaded training video files for this user."""
    d = training_video_dir(user_id)
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS)


def delete_training_videos(user_id: str) -> None:
    """Delete all uploaded training video files (called after training zip is committed)."""
    d = training_video_dir(user_id)
    for f in d.iterdir():
        if f.is_file():
            f.unlink(missing_ok=True)
    log.info(f"Training videos deleted for user={user_id}")


def delete_user_data(user_id: str) -> None:
    """GDPR: delete all stored data for this user."""
    import shutil
    from backend.services import blob_service

    user_dir = _root() / "users" / user_id
    if user_dir.exists():
        shutil.rmtree(user_dir)
    lora_dir = _root() / "loras" / user_id
    if lora_dir.exists():
        shutil.rmtree(lora_dir)
    if blob_service.is_enabled():
        blob_service.delete_user_videos(user_id)
    log.info(f"GDPR: deleted all data for user={user_id}")


def delete_generated_video(user_id: str, video_ref: str | None) -> bool:
    """Delete one generated video from local storage or blob storage."""
    if not video_ref:
        return False

    from backend.services import blob_service

    if video_ref.startswith("https://"):
        if blob_service.is_enabled():
            try:
                return blob_service.delete_video_by_url(video_ref)
            except Exception as exc:
                log.warning(f"Failed to delete blob video for user={user_id}: {exc}")
        return False

    outputs_dir = _root() / "users" / user_id / "outputs"
    safe_name = Path(video_ref).name
    p = outputs_dir / safe_name
    if p.exists() and p.is_file():
        p.unlink(missing_ok=True)
        return True
    return False
