"""Unit tests for replicate_service helpers (no real Replicate calls)."""
import io
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from backend.services import replicate_service


# ── _resolve_output ───────────────────────────────────────────────────────────

def test_resolve_output_returns_string_as_is():
    assert replicate_service._resolve_output("https://x.com/v.mp4") == "https://x.com/v.mp4"


def test_resolve_output_returns_last_item_from_list():
    items = ["https://x.com/1.mp4", "https://x.com/2.mp4"]
    assert replicate_service._resolve_output(items) == "https://x.com/2.mp4"


def test_resolve_output_materialises_generator():
    def _gen():
        yield "https://x.com/first.mp4"
        yield "https://x.com/last.mp4"
    assert replicate_service._resolve_output(_gen()) == "https://x.com/last.mp4"


def test_resolve_output_raises_on_empty_list():
    with pytest.raises(RuntimeError, match="empty output"):
        replicate_service._resolve_output([])


def test_resolve_output_passes_filelike_through():
    obj = MagicMock()
    obj.read.return_value = b"bytes"
    assert replicate_service._resolve_output(obj) is obj


# ── _download ─────────────────────────────────────────────────────────────────

def test_download_from_url(tmp_path):
    dest = tmp_path / "clip.mp4"

    def fake_urlretrieve(url, path):
        Path(path).write_bytes(b"fake-video-data")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve) as mock_retrieve:
        replicate_service._download("https://example.com/video.mp4", dest)
    mock_retrieve.assert_called_once_with("https://example.com/video.mp4", dest)


def test_download_from_filelike_object(tmp_path):
    dest = tmp_path / "clip.mp4"
    fake_data = b"video-bytes"
    fake_output = MagicMock()
    fake_output.read.return_value = fake_data

    replicate_service._download(fake_output, dest)

    assert dest.read_bytes() == fake_data


def test_download_creates_parent_dirs(tmp_path):
    dest = tmp_path / "a" / "b" / "clip.mp4"

    def fake_urlretrieve(url, path):
        Path(path).write_bytes(b"x")

    with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
        replicate_service._download("https://x.com/v.mp4", dest)
    assert dest.parent.exists()


def test_download_raises_on_empty_file(tmp_path):
    dest = tmp_path / "clip.mp4"

    with patch("urllib.request.urlretrieve", side_effect=lambda u, p: Path(p).write_bytes(b"")):
        with pytest.raises(RuntimeError, match="empty"):
            replicate_service._download("https://x.com/v.mp4", dest)


# ── _extract_last_frame ───────────────────────────────────────────────────────

def test_extract_last_frame_success(tmp_path):
    import numpy as np

    fake_frame = MagicMock()
    mock_cap = MagicMock()
    mock_cap.get.side_effect = lambda prop: {
        8: 30,   # CAP_PROP_FRAME_COUNT
    }.get(prop, 0)
    mock_cap.read.return_value = (True, fake_frame)

    out = tmp_path / "last.jpg"
    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.imwrite") as mock_write:
        result = replicate_service._extract_last_frame(tmp_path / "video.mp4", out)

    assert result == out
    mock_write.assert_called_once()
    out.parent.mkdir(parents=True, exist_ok=True)


def test_extract_last_frame_raises_on_bad_read(tmp_path):
    mock_cap = MagicMock()
    mock_cap.get.return_value = 10
    mock_cap.read.return_value = (False, None)

    out = tmp_path / "last.jpg"
    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="Could not extract last frame"):
            replicate_service._extract_last_frame(tmp_path / "video.mp4", out)


# ── test_connection ───────────────────────────────────────────────────────────

def test_test_connection_returns_true_on_success():
    mock_client = MagicMock()
    mock_client.models.get.return_value = MagicMock()
    with patch.object(replicate_service, "_client", return_value=mock_client):
        assert replicate_service.test_connection("r8_valid_key") is True


def test_test_connection_returns_false_on_exception():
    mock_client = MagicMock()
    mock_client.models.get.side_effect = Exception("Unauthorized")
    with patch.object(replicate_service, "_client", return_value=mock_client):
        assert replicate_service.test_connection("r8_bad_key") is False


# ── generate_clip_i2v ─────────────────────────────────────────────────────────

def test_generate_clip_i2v_requires_frame_or_clip(tmp_path):
    with pytest.raises(ValueError, match="Either first_frame_path or prev_clip_path"):
        replicate_service.generate_clip_i2v(
            api_key="r8_x",
            output_path=tmp_path / "out.mp4",
            prompt="test",
        )


def test_generate_clip_i2v_uses_first_frame(tmp_path):
    selfie = tmp_path / "selfie.jpg"
    selfie.write_bytes(b"jpeg")
    out = tmp_path / "out.mp4"

    mock_client = MagicMock()
    mock_client.run.return_value = "https://cdn.replicate.com/out.mp4"

    with patch.object(replicate_service, "_client", return_value=mock_client), \
         patch.object(replicate_service, "_download") as mock_dl:
        replicate_service.generate_clip_i2v(
            api_key="r8_x", output_path=out, prompt="walking", first_frame_path=selfie
        )

    from backend.core.config import get_settings
    call_kwargs = mock_client.run.call_args
    assert call_kwargs[0][0] == get_settings().replicate_i2v_model
    inp = call_kwargs[1]["input"]
    assert "first_frame" in inp
    assert "first_clip" not in inp
    mock_dl.assert_called_once()


def test_generate_clip_i2v_uses_prev_clip_for_continuation(tmp_path):
    prev = tmp_path / "prev.mp4"
    prev.write_bytes(b"mp4")
    out = tmp_path / "out.mp4"
    fake_frame = tmp_path / "prev_last.jpg"
    fake_frame.write_bytes(b"jpeg")

    mock_client = MagicMock()
    mock_client.run.return_value = "https://cdn.replicate.com/out.mp4"

    with patch.object(replicate_service, "_client", return_value=mock_client), \
         patch.object(replicate_service, "_extract_last_frame", return_value=fake_frame), \
         patch.object(replicate_service, "_download"):
        replicate_service.generate_clip_i2v(
            api_key="r8_x", output_path=out, prompt="walking", prev_clip_path=prev
        )

    inp = mock_client.run.call_args[1]["input"]
    assert "first_frame" in inp
    assert "first_clip" not in inp


# ── generate_clip_r2v ─────────────────────────────────────────────────────────

def test_generate_clip_r2v_raises_on_empty_refs(tmp_path):
    with pytest.raises(ValueError, match="At least one reference frame"):
        replicate_service.generate_clip_r2v(
            api_key="r8_x",
            output_path=tmp_path / "out.mp4",
            prompt="test",
            reference_frame_paths=[],
        )


def test_generate_clip_r2v_passes_reference_images(tmp_path):
    frames = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.jpg"
        p.write_bytes(b"jpeg")
        frames.append(p)
    out = tmp_path / "out.mp4"

    mock_client = MagicMock()
    mock_client.run.return_value = "https://cdn.replicate.com/out.mp4"

    with patch.object(replicate_service, "_client", return_value=mock_client), \
         patch.object(replicate_service, "_download"):
        replicate_service.generate_clip_r2v(
            api_key="r8_x", output_path=out, prompt="talking",
            reference_frame_paths=frames,
        )

    from backend.core.config import get_settings
    call_args = mock_client.run.call_args
    assert call_args[0][0] == get_settings().replicate_r2v_model
    inp = call_args[1]["input"]
    assert "reference_images" in inp
    assert len(inp["reference_images"]) == 3


# ── poll_lora_training ────────────────────────────────────────────────────────

def test_poll_lora_training_succeeded_returns_version_and_weights():
    mock_training = MagicMock()
    mock_training.status = "succeeded"
    mock_training.output = {
        "version": "owner/model:abc123",
        "weights": "https://cdn.replicate.com/weights.safetensors",
    }
    mock_training.error = None

    mock_client = MagicMock()
    mock_client.trainings.get.return_value = mock_training

    with patch.object(replicate_service, "_client", return_value=mock_client):
        result = replicate_service.poll_lora_training("r8_key", "train_id")

    assert result["status"] == "succeeded"
    assert result["version_id"] == "owner/model:abc123"
    assert result["weights_url"] == "https://cdn.replicate.com/weights.safetensors"
    assert result["error"] is None


def test_poll_lora_training_processing_returns_no_version():
    mock_training = MagicMock()
    mock_training.status = "processing"
    mock_training.output = None
    mock_training.error = None

    mock_client = MagicMock()
    mock_client.trainings.get.return_value = mock_training

    with patch.object(replicate_service, "_client", return_value=mock_client):
        result = replicate_service.poll_lora_training("r8_key", "train_id")

    assert result["status"] == "processing"
    assert result["version_id"] is None
    assert result["weights_url"] is None


def test_poll_lora_training_failed_returns_error():
    mock_training = MagicMock()
    mock_training.status = "failed"
    mock_training.output = None
    mock_training.error = "OOM on GPU"

    mock_client = MagicMock()
    mock_client.trainings.get.return_value = mock_training

    with patch.object(replicate_service, "_client", return_value=mock_client):
        result = replicate_service.poll_lora_training("r8_key", "train_id")

    assert result["status"] == "failed"
    assert result["error"] == "OOM on GPU"


# ── generate_clip_lora ────────────────────────────────────────────────────────

def test_generate_clip_lora_always_uses_destination_model(tmp_path):
    """Option B disabled: always use destination model (versioned ref), never base trainer.
    
    Reason: zsxkib/hunyuan-video-lora is a trainer-only model, not an inference model.
    Calling it with inference params produces empty/garbage output (green screen).
    """
    out = tmp_path / "clip.mp4"
    mock_client = MagicMock()
    mock_client.run.return_value = "https://cdn.replicate.com/clip.mp4"

    with patch.object(replicate_service, "_client", return_value=mock_client), \
         patch.object(replicate_service, "_download"):
        replicate_service.generate_clip_lora(
            api_key="r8_x",
            output_path=out,
            prompt="walking on beach",
            lora_model_ref="owner/model",
            lora_version_id="owner/model:abc123",
            trigger_word="SBJT01",
            lora_weights_url="https://cdn.replicate.com/weights.safetensors",  # provided but ignored
        )

    ref_used = mock_client.run.call_args[0][0]
    inp = mock_client.run.call_args[1]["input"]
    # Even with weights_url provided, should use destination model (not trainer)
    assert ref_used == "owner/model:abc123"
    assert "lora_weights" not in inp  # weights URL parameter should NOT be passed
    assert "SBJT01" in inp["prompt"]
    assert "walking on beach" in inp["prompt"]


def test_generate_clip_lora_uses_versioned_destination_model(tmp_path):
    """Always use the versioned destination model for LoRA inference (Option A only)."""
    out = tmp_path / "clip.mp4"
    mock_client = MagicMock()
    mock_client.run.return_value = "https://cdn.replicate.com/clip.mp4"

    with patch.object(replicate_service, "_client", return_value=mock_client), \
         patch.object(replicate_service, "_download"):
        replicate_service.generate_clip_lora(
            api_key="r8_x",
            output_path=out,
            prompt="dancing",
            lora_model_ref="owner/model",
            lora_version_id="owner/model:abc123",
            trigger_word="SBJT01",
            lora_weights_url=None,
        )

    ref_used = mock_client.run.call_args[0][0]
    inp = mock_client.run.call_args[1]["input"]
    assert ref_used == "owner/model:abc123"
    assert "lora_weights" not in inp


def test_generate_clip_lora_option_a_constructs_versioned_ref_from_bare_hash(tmp_path):
    """If version_id is just a hash (no colon), it should be prefixed with model_ref."""
    out = tmp_path / "clip.mp4"
    mock_client = MagicMock()
    mock_client.run.return_value = "https://cdn.replicate.com/clip.mp4"

    with patch.object(replicate_service, "_client", return_value=mock_client), \
         patch.object(replicate_service, "_download"):
        replicate_service.generate_clip_lora(
            api_key="r8_x",
            output_path=out,
            prompt="dancing",
            lora_model_ref="owner/model",
            lora_version_id="abc123hashonly",
            lora_weights_url=None,
        )

    ref_used = mock_client.run.call_args[0][0]
    assert ref_used == "owner/model:abc123hashonly"
