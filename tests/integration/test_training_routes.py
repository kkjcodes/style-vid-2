"""Integration tests for /api/v1/training routes."""
import uuid
import pytest
from unittest.mock import patch, MagicMock
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
        json={"username": f"trtest_{suffix}", "password": "Pass1234", "email": f"trtest_{suffix}@example.com"},
    )
    assert r.status_code == 201
    data = r.json()
    return {"headers": {"Authorization": f"Bearer {data['access_token']}"}, "user_id": data["user_id"]}


def _set_frames_job(user_id: str, job_id: str) -> None:
    from backend.db.database import SessionLocal
    from backend.db import crud

    db = SessionLocal()
    try:
        crud.set_frames_training_job(db, user_id, job_id)
    finally:
        db.close()


# ── POST /api/v1/training/start ───────────────────────────────────────────────

def test_start_training_requires_auth(client):
    r = client.post("/api/v1/training/start", json={"youtube_urls": ["https://youtube.com/watch?v=abc"]})
    assert r.status_code == 403


def test_start_training_requires_urls(client, auth):
    r = client.post("/api/v1/training/start", json={"youtube_urls": []}, headers=auth["headers"])
    assert r.status_code == 422
    assert "Upload videos first" in r.json()["detail"]


def test_start_training_accepts_many_urls_for_backward_compat(client, auth):
    r = client.post(
        "/api/v1/training/start",
        json={"youtube_urls": [f"https://youtube.com/watch?v={i}" for i in range(6)]},
        headers=auth["headers"],
    )
    assert r.status_code == 200


def test_start_training_queues_task(client, auth):
    with patch("backend.workers.training_worker.extract_reference_frames") as mock_fn:
        mock_fn.apply_async.return_value = MagicMock()
        r = client.post(
            "/api/v1/training/start",
            json={"youtube_urls": ["https://youtube.com/watch?v=abc123"], "max_frames": 15},
            headers=auth["headers"],
        )

    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "pending"


# ── GET /api/v1/training/status/{job_id} ─────────────────────────────────────

def test_training_status_pending(client, auth):
    _set_frames_job(auth["user_id"], "fake-job")
    mock_result = MagicMock()
    mock_result.state = "PENDING"
    mock_result.info = {}

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/training/status/fake-job", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["progress_pct"] == 0


def test_training_status_running(client, auth):
    _set_frames_job(auth["user_id"], "fake-job")
    mock_result = MagicMock()
    mock_result.state = "PROGRESS"
    mock_result.info = {"stage": "extracting", "progress_pct": 60}

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/training/status/fake-job", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"
    assert body["stage"] == "extracting"
    assert body["progress_pct"] == 60


def test_training_status_completed_with_frame_count(client, auth):
    _set_frames_job(auth["user_id"], "done-job")
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.info = {"stage": "done", "progress_pct": 100, "frame_count": 18}

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/training/status/done-job", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["frame_count"] == 18


def test_training_status_failed(client, auth):
    _set_frames_job(auth["user_id"], "bad-job")
    mock_result = MagicMock()
    mock_result.state = "FAILURE"
    mock_result.info = RuntimeError("yt-dlp: video unavailable")

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/training/status/bad-job", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert body["error"] is not None


def test_training_status_rejects_other_users_job(client):
    suffix_a = uuid.uuid4().hex[:8]
    reg_a = client.post(
        "/auth/register",
        json={"username": f"fjob_a_{suffix_a}", "password": "Pass1234", "email": f"fjob_a_{suffix_a}@example.com"},
    )
    user_a = reg_a.json()

    suffix_b = uuid.uuid4().hex[:8]
    reg_b = client.post(
        "/auth/register",
        json={"username": f"fjob_b_{suffix_b}", "password": "Pass1234", "email": f"fjob_b_{suffix_b}@example.com"},
    )
    user_b = reg_b.json()

    _set_frames_job(user_a["user_id"], "job-owner-a")
    headers_b = {"Authorization": f"Bearer {user_b['access_token']}"}

    r = client.get("/api/v1/training/status/job-owner-a", headers=headers_b)
    assert r.status_code == 403


# ── GET /api/v1/training/frames ──────────────────────────────────────────────

def test_list_frames_empty(client, auth, tmp_path, monkeypatch):
    import backend.services.storage_service as ss
    monkeypatch.setattr(ss.settings, "local_storage_dir", str(tmp_path))
    r = client.get("/api/v1/training/frames", headers=auth["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["frame_count"] == 0
    assert body["has_frames"] is False


def test_list_frames_with_data(client, auth, tmp_path, monkeypatch):
    import backend.services.storage_service as ss
    monkeypatch.setattr(ss.settings, "local_storage_dir", str(tmp_path))
    d = ss.reference_frames_dir(auth["user_id"])
    for i in range(5):
        (d / f"frame_{i:03d}.jpg").touch()

    r = client.get("/api/v1/training/frames", headers=auth["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["frame_count"] == 5
    assert body["has_frames"] is True
    assert len(body["frames"]) == 5


# ── DELETE /api/v1/training/frames ───────────────────────────────────────────

def test_delete_frames(client, auth, tmp_path, monkeypatch):
    import backend.services.storage_service as ss
    monkeypatch.setattr(ss.settings, "local_storage_dir", str(tmp_path))
    d = ss.reference_frames_dir(auth["user_id"])
    for i in range(3):
        (d / f"frame_{i:03d}.jpg").touch()

    r = client.delete("/api/v1/training/frames", headers=auth["headers"])
    assert r.status_code == 200
    assert r.json()["deleted"] == 3
    assert not ss.has_reference_frames(auth["user_id"])


def test_delete_frames_idempotent(client, auth, tmp_path, monkeypatch):
    import backend.services.storage_service as ss
    monkeypatch.setattr(ss.settings, "local_storage_dir", str(tmp_path))
    r = client.delete("/api/v1/training/frames", headers=auth["headers"])
    assert r.status_code == 200
    assert r.json()["deleted"] == 0


# ── GET /api/v1/training/lora/status/{job_id} ───────────────────────────────

def test_lora_status_rejects_other_users_job(client):
    suffix_a = uuid.uuid4().hex[:8]
    reg_a = client.post(
        "/auth/register",
        json={"username": f"lora_a_{suffix_a}", "password": "Pass1234", "email": f"lora_a_{suffix_a}@example.com"},
    )
    user_a = reg_a.json()

    suffix_b = uuid.uuid4().hex[:8]
    reg_b = client.post(
        "/auth/register",
        json={"username": f"lora_b_{suffix_b}", "password": "Pass1234", "email": f"lora_b_{suffix_b}@example.com"},
    )
    user_b = reg_b.json()

    from backend.db.database import SessionLocal
    from backend.db import crud
    db = SessionLocal()
    try:
        crud.set_lora_training_started(db, user_a["user_id"], "job-owner-a", "owner-a/model", "TRIGA")
    finally:
        db.close()

    headers_b = {"Authorization": f"Bearer {user_b['access_token']}"}
    r = client.get("/api/v1/training/lora/status/job-owner-a", headers=headers_b)
    assert r.status_code == 403


def test_lora_status_allows_owner_job(client):
    suffix = uuid.uuid4().hex[:8]
    reg = client.post(
        "/auth/register",
        json={"username": f"lora_o_{suffix}", "password": "Pass1234", "email": f"lora_o_{suffix}@example.com"},
    )
    user = reg.json()

    from backend.db.database import SessionLocal
    from backend.db import crud
    db = SessionLocal()
    try:
        crud.set_lora_celery_job(db, user["user_id"], "job-owner-ok")
    finally:
        db.close()

    mock_result = MagicMock()
    mock_result.state = "PENDING"
    mock_result.info = {}
    headers = {"Authorization": f"Bearer {user['access_token']}"}
    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/training/lora/status/job-owner-ok", headers=headers)

    assert r.status_code == 200
    assert r.json()["status"] == "pending"
