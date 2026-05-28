from __future__ import annotations
import base64
from datetime import datetime, timedelta
import bcrypt
from jose import jwt, JWTError
from backend.core.config import get_settings

settings = get_settings()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_access_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiry_hours)
    return jwt.encode(
        {"sub": user_id, "exp": expire},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def decode_token(token: str) -> str | None:
    """Return user_id (sub) from token, or None if invalid/expired."""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return payload.get("sub")
    except JWTError:
        return None


def _fernet():
    import hashlib
    from cryptography.fernet import Fernet
    # SHA-256 gives a consistent 32-byte key regardless of source length,
    # avoiding the truncation/padding weakness of direct slicing.
    raw = hashlib.sha256(settings.encryption_key.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(raw))


def encrypt_key(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt_key(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()


def hash_reset_token(token: str) -> str:
    """Hash a password-reset token for secure storage."""
    return bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()


def verify_reset_token(plain_token: str, hashed_token: str) -> bool:
    """Verify a password-reset token against its hash."""
    return bcrypt.checkpw(plain_token.encode(), hashed_token.encode())
