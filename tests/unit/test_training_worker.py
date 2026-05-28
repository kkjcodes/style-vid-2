"""Unit tests for training_worker helpers."""
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from backend.workers import training_worker


# ── _download_video ───────────────────────────────────────────────────────────

def test_download_video_returns_mp4_path(tmp_path):
    mp4 = tmp_path / "abc123.mp4"
    mp4.write_bytes(b"data")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = training_worker._download_video("https://youtube.com/watch?v=abc", tmp_path)

    assert result == mp4


def test_download_video_returns_none_on_failure(tmp_path):
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("yt-dlp", 600)):
        result = training_worker._download_video("https://youtube.com/watch?v=bad", tmp_path)
    assert result is None


def test_download_video_returns_none_when_no_mp4_created(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        # No .mp4 file created
        result = training_worker._download_video("https://youtube.com/watch?v=x", tmp_path)
    assert result is None


def test_download_video_uses_yt_dlp(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"x")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        training_worker._download_video("https://youtube.com/watch?v=abc", tmp_path)
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "yt-dlp"
    assert "--no-playlist" in cmd
    # Format must NOT contain [ext=mp4] — breaks YouTube Shorts
    format_idx = cmd.index("--format")
    assert "[ext=mp4]" not in cmd[format_idx + 1]


# ── _sample_frames ────────────────────────────────────────────────────────────

def test_sample_frames_returns_frame_paths(tmp_path):
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda p: {4: 24.0, 7: 240}.get(p, 0)  # fps=24, total=240

    frame_data = MagicMock()
    read_returns = [(True, frame_data)] * 240 + [(False, None)]
    mock_cap.read.side_effect = read_returns

    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.imwrite"):
        frames = training_worker._sample_frames(tmp_path / "v.mp4", tmp_path, sample_every_n_sec=2)

    # At 24fps, every 2s = every 48 frames. 240 frames → ~5 samples
    assert len(frames) == 5


def test_sample_frames_empty_video(tmp_path):
    mock_cap = MagicMock()
    mock_cap.get.return_value = 0
    mock_cap.read.return_value = (False, None)

    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.imwrite"):
        frames = training_worker._sample_frames(tmp_path / "v.mp4", tmp_path)

    assert frames == []


# ── _score_faces ─────────────────────────────────────────────────────────────

def test_score_faces_filters_low_score(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")

    mock_face = MagicMock()
    mock_face.det_score = 0.3  # below min threshold (0.65)
    mock_face.bbox = [10, 10, 50, 50]

    mock_app = MagicMock()
    mock_app.get.return_value = [mock_face]

    import numpy as np
    fake_img = MagicMock()
    fake_img.shape = (640, 480, 3)

    with patch("insightface.app.FaceAnalysis", return_value=mock_app), \
         patch("cv2.imread", return_value=fake_img):
        results = training_worker._score_faces([frame])

    assert results == []


def test_score_faces_keeps_high_score_frame(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")

    mock_face = MagicMock()
    mock_face.det_score = 0.92
    mock_face.bbox = [50, 50, 350, 450]  # large face
    mock_face.pose = MagicMock()
    mock_face.pose.__iter__ = lambda s: iter([5.0, 3.0, 0.0])

    import numpy as np
    mock_face.pose = np.array([5.0, 3.0, 0.0])

    mock_app = MagicMock()
    mock_app.get.return_value = [mock_face]

    fake_img = MagicMock()
    fake_img.shape = (640, 480, 3)

    with patch("insightface.app.FaceAnalysis", return_value=mock_app), \
         patch("cv2.imread", return_value=fake_img):
        results = training_worker._score_faces([frame])

    assert len(results) == 1
    score, path = results[0]
    assert score > 0.5
    assert path == frame


def test_score_faces_skips_unreadable_image(tmp_path):
    frame = tmp_path / "bad.jpg"
    frame.write_bytes(b"not-an-image")

    mock_app = MagicMock()

    with patch("insightface.app.FaceAnalysis", return_value=mock_app), \
         patch("cv2.imread", return_value=None):
        results = training_worker._score_faces([frame])

    assert results == []
    mock_app.get.assert_not_called()


def test_score_faces_selects_largest_face_when_multiple(tmp_path):
    """When multiple faces present, best face (largest bbox) should be used."""
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")

    import numpy as np

    small_face = MagicMock()
    small_face.det_score = 0.8
    small_face.bbox = [10, 10, 40, 40]   # small
    small_face.pose = np.zeros(3)

    large_face = MagicMock()
    large_face.det_score = 0.9
    large_face.bbox = [50, 50, 300, 400]  # large
    large_face.pose = np.zeros(3)

    mock_app = MagicMock()
    mock_app.get.return_value = [small_face, large_face]

    fake_img = MagicMock()
    fake_img.shape = (480, 640, 3)

    with patch("insightface.app.FaceAnalysis", return_value=mock_app), \
         patch("cv2.imread", return_value=fake_img):
        results = training_worker._score_faces([frame])

    assert len(results) == 1


# ── _normalize_video_for_processing ─────────────────────────────────────────

def test_normalize_video_transcodes_av1(tmp_path):
    src = tmp_path / "input_av1.mp4"
    src.write_bytes(b"source")
    out = tmp_path / "tmp" / "normalized" / "input_av1_h264.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"converted")

    probe_result = MagicMock(returncode=0, stdout='{"streams":[{"codec_name":"av1"}]}')
    ffmpeg_result = MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=[probe_result, ffmpeg_result]) as mock_run:
        normalized = training_worker._normalize_video_for_processing(src, tmp_path / "tmp", "u1")

    assert normalized == out
    ffmpeg_cmd = mock_run.call_args_list[1][0][0]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert "libx264" in ffmpeg_cmd


def test_normalize_video_skips_non_av1(tmp_path):
    src = tmp_path / "input_h264.mp4"
    src.write_bytes(b"source")

    probe_result = MagicMock(returncode=0, stdout='{"streams":[{"codec_name":"h264"}]}')

    with patch("subprocess.run", return_value=probe_result) as mock_run:
        normalized = training_worker._normalize_video_for_processing(src, tmp_path / "tmp", "u1")

    assert normalized == src
    assert mock_run.call_count == 1


def test_normalize_video_falls_back_on_ffmpeg_failure(tmp_path):
    src = tmp_path / "input_av1.mp4"
    src.write_bytes(b"source")

    probe_result = MagicMock(returncode=0, stdout='{"streams":[{"codec_name":"av1"}]}')
    ffmpeg_result = MagicMock(returncode=1, stdout="", stderr="decode failed")

    with patch("subprocess.run", side_effect=[probe_result, ffmpeg_result]):
        normalized = training_worker._normalize_video_for_processing(src, tmp_path / "tmp", "u1")

    assert normalized == src
