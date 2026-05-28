"""Unit tests for Settings / config."""
from backend.core.config import get_settings, Settings


def test_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_required_model_ids():
    s = get_settings()
    assert "wan-video" in s.replicate_i2v_model
    assert "wan-video" in s.replicate_r2v_model


def test_clip_duration_sane():
    s = get_settings()
    assert 2 <= s.replicate_clip_duration <= 15


def test_resolution_valid():
    s = get_settings()
    assert s.replicate_resolution in ("480p", "720p", "1080p")


def test_aspect_ratio_valid():
    s = get_settings()
    assert s.replicate_aspect_ratio in ("16:9", "9:16", "1:1")


def test_min_face_score_range():
    s = get_settings()
    assert 0.0 < s.min_face_score < 1.0


def test_max_reference_frames_sane():
    s = get_settings()
    assert 5 <= s.max_reference_frames <= 50


def test_default_clips_per_video():
    s = get_settings()
    assert s.replicate_clips_per_video >= 1


def test_redis_url_format():
    s = get_settings()
    assert s.redis_url.startswith("redis://") or s.redis_url.startswith("rediss://")


def test_local_storage_dir_nonempty():
    s = get_settings()
    assert s.local_storage_dir
