"""
Integration tests for /api/v1/pipeline routes.
Uses FastAPI TestClient — no real Replicate calls, Celery tasks are mocked.
"""
import io
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
    """Register a unique test user and return token + user_id."""
    suffix = uuid.uuid4().hex[:8]
    r = client.post(
        "/auth/register",
        json={"username": f"ptest_{suffix}", "password": "Pass1234", "email": f"ptest_{suffix}@example.com"},
    )
    assert r.status_code == 201
    data = r.json()
    return {"headers": {"Authorization": f"Bearer {data['access_token']}"}, "user_id": data["user_id"]}


@pytest.fixture()
def auth_with_key(client, auth):
    """Auth fixture that also stores a mock Replicate key."""
    with patch("backend.services.replicate_service.test_connection", return_value=True):
        client.put("/auth/replicate-key", json={"replicate_key": "r8_testkey"}, headers=auth["headers"])
    return auth


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# ── POST /api/v1/pipeline/test-key ───────────────────────────────────────────

def test_test_key_invalid(client):
    with patch("backend.services.replicate_service.test_connection", return_value=False):
        r = client.post("/api/v1/pipeline/test-key", json={"replicate_key": "r8_bad"})
    assert r.status_code == 401
    assert "Invalid" in r.json()["detail"]


def test_test_key_valid(client):
    with patch("backend.services.replicate_service.test_connection", return_value=True):
        r = client.post("/api/v1/pipeline/test-key", json={"replicate_key": "r8_valid"})
    assert r.status_code == 200
    assert r.json()["valid"] is True


# ── POST /api/v1/pipeline/selfie ─────────────────────────────────────────────

def test_selfie_no_face_detected(client, auth):
    with patch("backend.api.routes.pipeline.identity_service.extract_and_save", return_value=False), \
         patch("backend.api.routes.pipeline.storage_service.selfie_path") as mock_path:
        mock_path.return_value.__truediv__ = MagicMock()
        import tempfile, pathlib
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp = pathlib.Path(f.name)
        mock_path.return_value = tmp

        r = client.post(
            "/api/v1/pipeline/selfie",
            files={"selfie": ("selfie.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
            headers=auth["headers"],
        )

    assert r.status_code == 422
    assert "face" in r.json()["detail"].lower()


def test_selfie_face_detected(client, auth, tmp_path):
    dest = tmp_path / "selfie.jpg"
    with patch("backend.api.routes.pipeline.storage_service.selfie_path", return_value=dest), \
         patch("backend.api.routes.pipeline.identity_service.extract_and_save", return_value=True):
        r = client.post(
            "/api/v1/pipeline/selfie",
            files={"selfie": ("selfie.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
            headers=auth["headers"],
        )

    assert r.status_code == 200
    assert r.json()["face_detected"] is True


# ── POST /api/v1/pipeline/generate ───────────────────────────────────────────

def test_generate_requires_replicate_key(client, auth, tmp_path):
    selfie = tmp_path / "selfie.jpg"
    selfie.write_bytes(b"x")
    with patch("backend.api.routes.pipeline.storage_service.selfie_path", return_value=selfie):
        r = client.post(
            "/api/v1/pipeline/generate",
            json={"prompt": "walking in a park"},
            headers=auth["headers"],
        )
    assert r.status_code == 422
    assert "Replicate" in r.json()["detail"]


def test_generate_requires_selfie(client, auth_with_key, tmp_path):
    with patch("backend.api.routes.pipeline.storage_service.selfie_path",
               return_value=tmp_path / "nonexistent.jpg"):
        r = client.post(
            "/api/v1/pipeline/generate",
            json={"prompt": "walking in a park"},
            headers=auth_with_key["headers"],
        )
    assert r.status_code == 422
    assert "selfie" in r.json()["detail"].lower()


def test_generate_queues_task(client, auth_with_key, tmp_path):
    selfie = tmp_path / "selfie.jpg"
    selfie.write_bytes(b"x")

    with patch("backend.api.routes.pipeline.storage_service.selfie_path", return_value=selfie), \
         patch("backend.workers.pipeline_worker.run_pipeline") as mock_run:
        mock_run.apply_async.return_value = MagicMock()
        r = client.post(
            "/api/v1/pipeline/generate",
            json={"prompt": "walking in NYC at sunset", "num_clips": 2, "resolution": "720p"},
            headers=auth_with_key["headers"],
        )

    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "pending"
    assert body["video_duration_sec"] == 10  # 2 clips × 5s
    queued_kwargs = mock_run.apply_async.call_args.kwargs["kwargs"]
    assert "replicate_key" not in queued_kwargs


def test_generate_requires_auth(client):
    r = client.post("/api/v1/pipeline/generate", json={"prompt": "test"})
    assert r.status_code == 403


# ── GET /api/v1/pipeline/jobs/{job_id} ───────────────────────────────────────

def test_job_status_pending(client, auth):
    mock_result = MagicMock()
    mock_result.state = "PENDING"
    mock_result.info = {}

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/pipeline/jobs/fake-job-id", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending"
    assert body["progress_pct"] == 0


def test_job_status_running(client, auth):
    mock_result = MagicMock()
    mock_result.state = "PROGRESS"
    mock_result.info = {"progress_pct": 45, "message": "Generating clip 2/3…"}

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/pipeline/jobs/fake-job-id", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "running"
    assert body["progress_pct"] == 45
    assert "2/3" in body["message"]


def test_job_status_completed(client, auth):
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.info = {
        "progress_pct": 100,
        "message": "Video ready!",
        "video_path": "/tmp/stylevid2/users/x/outputs/job123.mp4",
    }

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/pipeline/jobs/job123", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["video_url"] is not None


def test_job_status_failed(client, auth):
    mock_result = MagicMock()
    mock_result.state = "FAILURE"
    mock_result.info = Exception("Replicate rate limited")

    with patch("backend.workers.celery_app.celery_app.AsyncResult", return_value=mock_result):
        r = client.get("/api/v1/pipeline/jobs/bad-job", headers=auth["headers"])

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert body["error"] is not None


# ── GET /api/v1/pipeline/history ─────────────────────────────────────────────

def test_history_empty(client, auth):
    r = client.get("/api/v1/pipeline/history", headers=auth["headers"])
    assert r.status_code == 200
    assert r.json()["jobs"] == []


def test_history_requires_auth(client):
    r = client.get("/api/v1/pipeline/history")
    assert r.status_code == 403


# ── DELETE /api/v1/pipeline/history/{job_id} ───────────────────────────────

def test_delete_history_item(client, auth_with_key, tmp_path):
    selfie = tmp_path / "selfie.jpg"
    selfie.write_bytes(b"x")

    with patch("backend.api.routes.pipeline.storage_service.selfie_path", return_value=selfie), \
         patch("backend.workers.pipeline_worker.run_pipeline") as mock_run:
        mock_run.apply_async.return_value = MagicMock()
        start = client.post(
            "/api/v1/pipeline/generate",
            json={"prompt": "delete me", "num_clips": 1, "resolution": "720p"},
            headers=auth_with_key["headers"],
        )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    with patch("backend.api.routes.pipeline.storage_service.delete_generated_video", return_value=True):
        r = client.delete(f"/api/v1/pipeline/history/{job_id}", headers=auth_with_key["headers"])

    assert r.status_code == 200
    assert r.json()["deleted"] is True
    history = client.get("/api/v1/pipeline/history", headers=auth_with_key["headers"])
    assert history.status_code == 200
    assert all(j["job_id"] != job_id for j in history.json().get("jobs", []))


def test_delete_history_item_requires_owner(client, auth, tmp_path):
    # Create another user that owns the job.
    suffix = uuid.uuid4().hex[:8]
    r = client.post(
        "/auth/register",
        json={
            "username": f"owner_{suffix}",
            "password": "Pass1234",
            "email": f"owner_{suffix}@example.com",
        },
    )
    owner = r.json()
    owner_id = owner["user_id"]

    with patch("backend.services.replicate_service.test_connection", return_value=True):
        client.put(
            "/auth/replicate-key",
            json={"replicate_key": "r8_ownerkey"},
            headers={"Authorization": f"Bearer {owner['access_token']}"},
        )

    selfie = tmp_path / "owner_selfie.jpg"
    selfie.write_bytes(b"x")
    with patch("backend.api.routes.pipeline.storage_service.selfie_path", return_value=selfie), \
         patch("backend.workers.pipeline_worker.run_pipeline") as mock_run:
        mock_run.apply_async.return_value = MagicMock()
        start = client.post(
            "/api/v1/pipeline/generate",
            json={"prompt": "owner prompt", "num_clips": 1, "resolution": "720p"},
            headers={"Authorization": f"Bearer {owner['access_token']}"},
        )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    r = client.delete(f"/api/v1/pipeline/history/{job_id}", headers=auth["headers"])
    assert r.status_code == 403
