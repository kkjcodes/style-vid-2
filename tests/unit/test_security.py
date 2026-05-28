"""Unit tests for backend.core.security."""
import pytest
from unittest.mock import patch
from backend.core.security import (
    hash_password, verify_password,
    create_access_token, decode_token,
    encrypt_key, decrypt_key,
)


# ── Password hashing ──────────────────────────────────────────────────────────

def test_hash_password_returns_bcrypt_string():
    h = hash_password("mysecret")
    assert h.startswith("$2b$")


def test_verify_password_correct():
    h = hash_password("correct")
    assert verify_password("correct", h) is True


def test_verify_password_wrong():
    h = hash_password("correct")
    assert verify_password("wrong", h) is False


def test_verify_password_empty_plain():
    h = hash_password("nonempty")
    assert verify_password("", h) is False


# ── JWT ───────────────────────────────────────────────────────────────────────

def test_create_and_decode_token():
    token = create_access_token("user-123")
    assert decode_token(token) == "user-123"


def test_decode_invalid_token_returns_none():
    assert decode_token("not.a.token") is None


def test_decode_tampered_token_returns_none():
    token = create_access_token("user-abc")
    tampered = token[:-4] + "xxxx"
    assert decode_token(tampered) is None


def test_decode_token_different_users():
    t1 = create_access_token("user-1")
    t2 = create_access_token("user-2")
    assert decode_token(t1) == "user-1"
    assert decode_token(t2) == "user-2"


# ── Encryption ────────────────────────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip():
    plaintext = "r8_supersecretkey"
    cipher = encrypt_key(plaintext)
    assert cipher != plaintext
    assert decrypt_key(cipher) == plaintext


def test_encrypt_produces_different_ciphertext_each_time():
    k = "r8_key"
    assert encrypt_key(k) != encrypt_key(k)  # Fernet uses random IV


def test_decrypt_wrong_ciphertext_raises():
    from cryptography.fernet import InvalidToken
    with pytest.raises((InvalidToken, Exception)):
        decrypt_key("not-valid-ciphertext")
