"""
Auth routes.

POST /auth/register            create account
POST /auth/login               get JWT token
GET  /auth/me                  current user profile
PUT  /auth/replicate-key       save / update Replicate API key (encrypted at rest)
POST /auth/forgot-password     send password-reset email
POST /auth/reset-password      apply new password using reset token
"""
import logging
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from backend.api.limiter import limiter

from backend.db.database import get_db
from backend.db import crud
from backend.db.models import User
from backend.core.security import (
    verify_password, create_access_token, encrypt_key, hash_password,
    hash_reset_token, verify_reset_token,
)
from backend.api.deps import get_current_user

log = logging.getLogger("routes.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username may only contain letters, numbers, - and _.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Enter a valid email address.")
        return v


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _pw(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class ReplicateKeyRequest(BaseModel):
    replicate_key: str


def _token_response(user: User) -> dict:
    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
    }


@router.post("/register", status_code=201)
@limiter.limit("5/minute")
def register(request: Request, body: RegisterRequest, db: Session = Depends(get_db)):
    if crud.get_user_by_username(db, body.username):
        raise HTTPException(status_code=409, detail="Username already taken.")
    if crud.get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="An account with that email already exists.")
    user = crud.create_user(db, body.username, body.password, body.email)
    log.info(f"Registered: {user.username} ({user.id})")
    return _token_response(user)


@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, body.username)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return _token_response(user)


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "has_replicate_key": current_user.replicate_key_encrypted is not None,
    }


@router.post("/forgot-password")
@limiter.limit("3/minute")
def forgot_password(request: Request, body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Send a password-reset link to the email address.
    Always returns 200 regardless of whether the email exists (prevents enumeration).
    """
    from backend.services.email_service import send_reset_email

    user = crud.get_user_by_email(db, body.email.strip().lower())
    if user and user.is_active:
        token = secrets.token_hex(32)
        hashed = hash_reset_token(token)
        expires = datetime.utcnow() + timedelta(hours=1)
        crud.set_reset_token(db, user.id, hashed, expires)
        try:
            send_reset_email(user.email, token)
        except Exception as exc:
            log.error(f"Reset email failed for {user.email}: {exc}")
            # Don't surface the error to the caller — prevents enumeration
    else:
        log.info(f"Forgot-password request for unknown/inactive email: {body.email}")

    return {"message": "If that email is registered you will receive a reset link shortly."}


@router.post("/reset-password")
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Apply a new password using the token from the reset email."""
    from sqlalchemy import text

    # Find user by email lookup (prevent timing attacks on token format).
    # Token format is unforgeable due to secure random generation, so we scan all users.
    # In production with many users, consider indexing by reset_token hash prefix.
    users = db.query(User).filter(User.reset_token != None).all()
    user = None
    for u in users:
        if u.reset_token and verify_reset_token(body.token, u.reset_token):
            user = u
            break
    
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
    if not user.reset_token_expires or datetime.utcnow() > user.reset_token_expires:
        crud.clear_reset_token(db, user.id)
        raise HTTPException(status_code=400, detail="Reset token has expired. Request a new one.")

    crud.update_password(db, user.id, hash_password(body.new_password))
    crud.clear_reset_token(db, user.id)
    log.info(f"Password reset for user={user.username}")
    return {"message": "Password updated. You can now log in."}


@router.put("/replicate-key")
def set_replicate_key(
    body: ReplicateKeyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Validate and store the user's Replicate API key (encrypted at rest)."""
    from backend.services.replicate_service import test_connection

    if not body.replicate_key.startswith("r8_"):
        raise HTTPException(status_code=422, detail="Replicate keys start with r8_")
    if not test_connection(body.replicate_key):
        raise HTTPException(status_code=422, detail="Could not verify key with Replicate — check it and try again.")

    crud.set_replicate_key(db, current_user.id, encrypt_key(body.replicate_key))
    log.info(f"Replicate key updated: user={current_user.username}")
    return {"message": "Replicate API key saved."}
