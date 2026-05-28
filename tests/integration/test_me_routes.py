"""Integration tests for /api/v1/me routes."""
import uuid
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture()
def auth(client):
    suffix = uuid.uuid4().hex[:8]
    r = client.post(
        "/auth/register",
        json={"username": f"metest_{suffix}", "password": "Pass1234", "email": f"metest_{suffix}@example.com"},
    )
    assert r.status_code == 201
    data = r.json()
    return {"headers": {"Authorization": f"Bearer {data['access_token']}"}, "user_id": data["user_id"]}


# ── GET /api/v1/me ────────────────────────────────────────────────────────────

def test_get_profile_requires_auth(client):
    r = client.get("/api/v1/me")
    assert r.status_code == 403


def test_get_profile_returns_user_data(client, auth):
    r = client.get("/api/v1/me", headers=auth["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == auth["user_id"]
    assert "username" in body
    assert "has_selfie" in body
    assert "reference_frames" in body
    assert "generated_videos" in body


def test_get_profile_no_selfie_initially(client, auth):
    r = client.get("/api/v1/me", headers=auth["headers"])
    assert r.json()["has_selfie"] is False


def test_get_profile_no_replicate_key_initially(client, auth):
    r = client.get("/api/v1/me", headers=auth["headers"])
    assert r.json()["has_replicate_key"] is False


# ── DELETE /api/v1/me ─────────────────────────────────────────────────────────

def test_delete_account_requires_auth(client):
    r = client.delete("/api/v1/me")
    assert r.status_code == 403


def test_delete_account_wipes_data(client, auth):
    r = client.delete("/api/v1/me", headers=auth["headers"])
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    # Token should now be invalid (user no longer in DB)
    r2 = client.get("/api/v1/me", headers=auth["headers"])
    assert r2.status_code == 401
