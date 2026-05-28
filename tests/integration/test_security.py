"""Security integration tests — path traversal, ownership, auth boundaries."""
import uuid
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _register(client):
    suffix = uuid.uuid4().hex[:8]
    r = client.post(
        "/auth/register",
        json={"username": f"sec_{suffix}", "password": "Pass1234", "email": f"sec_{suffix}@example.com"},
    )
    assert r.status_code == 201
    d = r.json()
    return {"headers": {"Authorization": f"Bearer {d['access_token']}"}, "user_id": d["user_id"]}


# ── Auth boundaries ───────────────────────────────────────────────────────────

def test_unauthenticated_routes_return_403(client):
    protected = [
        ("POST",   "/api/v1/pipeline/selfie"),
        ("POST",   "/api/v1/pipeline/generate"),
        ("GET",    "/api/v1/pipeline/jobs/any-id"),
        ("GET",    "/api/v1/pipeline/history"),
        ("GET",    "/api/v1/pipeline/video/any.mp4"),
        ("POST",   "/api/v1/training/start"),
        ("GET",    "/api/v1/training/status/any-id"),
        ("GET",    "/api/v1/training/frames"),
        ("DELETE", "/api/v1/training/frames"),
        ("GET",    "/api/v1/me"),
        ("DELETE", "/api/v1/me"),
        ("GET",    "/auth/me"),
        ("PUT",    "/auth/replicate-key"),
    ]
    for method, path in protected:
        r = client.request(method, path)
        assert r.status_code in (403, 422), f"{method} {path} should require auth, got {r.status_code}"


def test_invalid_token_returns_401(client):
    headers = {"Authorization": "Bearer this.is.fake"}
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 401


# ── Job ownership ─────────────────────────────────────────────────────────────

def test_cannot_poll_other_users_job(client):
    user_a = _register(client)
    user_b = _register(client)

    # Submit a job as user_a
    import uuid as _uuid
    job_id = str(_uuid.uuid4())

    # Insert directly into the DB so we have a known job owned by user_a
    from backend.db.database import SessionLocal
    from backend.db import crud
    db = SessionLocal()
    try:
        crud.create_video_job(db, job_id, user_a["user_id"], "test prompt", "720p", 2)
    finally:
        db.close()

    # user_b tries to poll user_a's job
    mock_result = MagicMock()
    mock_result.state = "PENDING"
    mock_result.info = {}
    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get(f"/api/v1/pipeline/jobs/{job_id}", headers=user_b["headers"])

    assert r.status_code == 403


def test_can_poll_own_job(client):
    user = _register(client)
    job_id = str(uuid.uuid4())

    from backend.db.database import SessionLocal
    from backend.db import crud
    db = SessionLocal()
    try:
        crud.create_video_job(db, job_id, user["user_id"], "test prompt", "720p", 2)
    finally:
        db.close()

    mock_result = MagicMock()
    mock_result.state = "PENDING"
    mock_result.info = {}
    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get(f"/api/v1/pipeline/jobs/{job_id}", headers=user["headers"])

    assert r.status_code == 200


# ── Path traversal ────────────────────────────────────────────────────────────

def test_path_traversal_rejected(client):
    user = _register(client)
    traversal_filenames = [
        "../other_user/outputs/video.mp4",
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "valid_name/../other",
    ]
    for fname in traversal_filenames:
        r = client.get(f"/api/v1/pipeline/video/{fname}", headers=user["headers"])
        assert r.status_code in (400, 404), f"Expected 400/404 for '{fname}', got {r.status_code}"


def test_video_serve_rejects_missing_file(client):
    user = _register(client)
    r = client.get("/api/v1/pipeline/video/nonexistent.mp4", headers=user["headers"])
    assert r.status_code == 404


# ── Data isolation ────────────────────────────────────────────────────────────

def test_history_only_shows_own_jobs(client):
    user_a = _register(client)
    user_b = _register(client)

    from backend.db.database import SessionLocal
    from backend.db import crud
    db = SessionLocal()
    try:
        crud.create_video_job(db, str(uuid.uuid4()), user_a["user_id"], "user_a prompt", "720p", 2)
    finally:
        db.close()

    # user_b's history should be empty (their own jobs only)
    r = client.get("/api/v1/pipeline/history", headers=user_b["headers"])
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    # None of user_a's jobs should appear
    for j in jobs:
        assert j["prompt"] != "user_a prompt"


def test_profile_scoped_to_own_user(client):
    user_a = _register(client)
    user_b = _register(client)
    # Both should get their own profile, not each other's
    r_a = client.get("/api/v1/me", headers=user_a["headers"])
    r_b = client.get("/api/v1/me", headers=user_b["headers"])
    assert r_a.json()["user_id"] == user_a["user_id"]
    assert r_b.json()["user_id"] == user_b["user_id"]
    assert r_a.json()["user_id"] != r_b.json()["user_id"]
