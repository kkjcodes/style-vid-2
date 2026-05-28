"""Unit tests for video_chain_service.stitch."""
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.services import video_chain_service


def test_stitch_raises_on_empty_clips(tmp_path):
    with pytest.raises(ValueError, match="No clips"):
        video_chain_service.stitch([], tmp_path / "out.mp4")


def test_stitch_single_clip_copies(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake-video-data")
    out = tmp_path / "out.mp4"

    result = video_chain_service.stitch([src], out)

    assert result == out
    assert out.exists()
    assert out.read_bytes() == b"fake-video-data"


def test_stitch_multi_clips_calls_ffmpeg(tmp_path):
    clips = []
    for i in range(3):
        p = tmp_path / f"clip_{i}.mp4"
        p.write_bytes(b"data")
        clips.append(p)
    out = tmp_path / "out.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 0

    # Make ffmpeg "succeed" by also creating the output file
    def fake_run(cmd, **kwargs):
        out.write_bytes(b"stitched")
        return mock_result

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        video_chain_service.stitch(clips, out)

    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "ffmpeg" in call_args
    assert "-f" in call_args
    assert "concat" in call_args
    assert str(out) in call_args


def test_stitch_raises_on_ffmpeg_failure(tmp_path):
    clips = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for c in clips:
        c.write_bytes(b"x")
    out = tmp_path / "out.mp4"

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "ffmpeg: error opening output file"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="ffmpeg stitch failed"):
            video_chain_service.stitch(clips, out)


def test_stitch_creates_parent_dirs(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"data")
    out = tmp_path / "deep" / "nested" / "out.mp4"

    video_chain_service.stitch([src], out)

    assert out.exists()
