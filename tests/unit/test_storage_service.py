"""Unit tests for storage_service path helpers and GDPR deletion."""
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

from backend.services import storage_service


@pytest.fixture()
def tmp_storage(tmp_path, monkeypatch):
    """Redirect all storage paths to a temp directory for the test."""
    monkeypatch.setattr(storage_service.settings, "local_storage_dir", str(tmp_path))
    # Clear lru_cache so _root() picks up the monkeypatched value
    return tmp_path


def test_selfie_path_creates_dirs(tmp_storage):
    p = storage_service.selfie_path("alice")
    assert p.parent.exists()
    assert p.name == "selfie.jpg"
    assert "alice" in str(p)


def test_output_video_path(tmp_storage):
    p = storage_service.output_video_path("bob", "job-123")
    assert p.name == "job-123.mp4"
    assert p.parent.exists()


def test_reference_frames_dir_creates(tmp_storage):
    d = storage_service.reference_frames_dir("carol")
    assert d.exists()
    assert "reference_frames" in str(d)


def test_reference_frame_paths_empty(tmp_storage):
    frames = storage_service.reference_frame_paths("dave")
    assert frames == []


def test_has_reference_frames_false_when_empty(tmp_storage):
    assert storage_service.has_reference_frames("eve") is False


def test_has_reference_frames_true_when_files_exist(tmp_storage):
    d = storage_service.reference_frames_dir("frank")
    (d / "frame_000.jpg").touch()
    (d / "frame_001.jpg").touch()
    assert storage_service.has_reference_frames("frank") is True


def test_reference_frame_paths_sorted(tmp_storage):
    d = storage_service.reference_frames_dir("grace")
    (d / "frame_002.jpg").touch()
    (d / "frame_000.jpg").touch()
    (d / "frame_001.jpg").touch()
    paths = storage_service.reference_frame_paths("grace")
    names = [p.name for p in paths]
    assert names == sorted(names)


def test_delete_user_data_removes_user_dir(tmp_storage):
    uid = "henry"
    selfie = storage_service.selfie_path(uid)
    selfie.touch()
    assert selfie.exists()

    storage_service.delete_user_data(uid)

    assert not selfie.exists()
    assert not (tmp_storage / "users" / uid).exists()


def test_delete_user_data_idempotent(tmp_storage):
    """Deleting a non-existent user should not raise."""
    storage_service.delete_user_data("nonexistent_user_xyz")


def test_delete_user_data_removes_reference_frames(tmp_storage):
    uid = "iris"
    d = storage_service.reference_frames_dir(uid)
    (d / "frame_000.jpg").touch()
    assert storage_service.has_reference_frames(uid)

    storage_service.delete_user_data(uid)

    assert not d.exists()
