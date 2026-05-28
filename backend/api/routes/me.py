"""
User profile + GDPR routes.

GET    /api/v1/me          current user's data summary (JWT required)
DELETE /api/v1/me          GDPR erasure — wipes everything (JWT required)
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.core.logging_config import setup_logging
from backend.db import crud
from backend.db.database import get_db
from backend.db.models import User
from backend.services import storage_service

setup_logging()
log = logging.getLogger("routes.me")
router = APIRouter(prefix="/api/v1/me", tags=["me"])


@router.get("")
def get_profile(current_user: User = Depends(get_current_user)):
    """Return what data is stored for the current user."""
    selfie_path = storage_service.selfie_path(current_user.id)
    ref_frames = storage_service.reference_frame_paths(current_user.id)
    output_dir = storage_service.output_video_path(current_user.id, "_").parent

    videos = list(output_dir.glob("*.mp4")) if output_dir.exists() else []
    return {
        "user_id":          current_user.id,
        "username":         current_user.username,
        "has_selfie":       selfie_path.exists(),
        "reference_frames": len(ref_frames),
        "generated_videos": len(videos),
        "has_replicate_key": current_user.replicate_key_encrypted is not None,
    }


@router.delete("")
def delete_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """GDPR Article 17 — Right to erasure. Deletes all files and DB records."""
    storage_service.delete_user_data(current_user.id)
    crud.delete_user(db, current_user.id)
    log.info(f"GDPR erasure complete for user={current_user.username} ({current_user.id})")
    return {"deleted": True, "message": "All your data has been permanently deleted."}
