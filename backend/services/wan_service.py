"""
Wan2.1 / Wan2.2 inference service.

Track A — WanImageToVideoPipeline (Wan2.1-I2V):
  selfie image + text prompt → video

Track B — WanAnimatePipeline (Wan2.2-Animate):
  selfie image + preprocessed pose/face videos → body-animated video

Confirmed API from diffusers docs (not guessed):
  - WanImageToVideoPipeline.__call__(image, prompt, negative_prompt,
      height, width, num_frames, guidance_scale) → .frames[0]
  - WanAnimatePipeline.__call__(image, pose_video, face_video, prompt,
      negative_prompt, height, width, guidance_scale, mode) → .frames[0]
  - VAE must be torch.float32; image_encoder must be torch.float32
  - Transformer uses bfloat16 + PYTORCH_ENABLE_MPS_FALLBACK=1
  - Image resize: mod_value = vae_scale_factor_spatial * patch_size[1]
  - num_frames must satisfy 4k+1 (e.g. 81 = 4*20+1)
  - export_to_video(frames, path, fps=16) for output
"""
from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from backend.core.config import get_settings

log = logging.getLogger("wan_service")
settings = get_settings()


# ─── Memory helpers ───────────────────────────────────────────────────────────

def _mem_str() -> str:
    """Current process RSS + MPS allocator stats as a human-readable string."""
    parts: list[str] = []
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
        parts.append(f"RSS={rss/1e9:.2f}GB")
    except Exception:
        import resource
        # ru_maxrss on macOS is peak RSS in bytes
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        parts.append(f"peak_RSS={peak/1e9:.2f}GB")

    if torch.backends.mps.is_available():
        try:
            alloc  = torch.mps.current_allocated_memory()
            driver = torch.mps.driver_allocated_memory()
            parts.append(f"MPS_alloc={alloc/1e9:.2f}GB MPS_driver={driver/1e9:.2f}GB")
        except Exception:
            pass

    return "  ".join(parts)


def _mem(label: str) -> None:
    log.info(f"[MEM] {label}: {_mem_str()}")

# ─── Device ───────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _device()
# VAE and image_encoder are always float32 (confirmed from docs).
# Transformer uses bfloat16 — supported on MPS with PYTORCH_ENABLE_MPS_FALLBACK=1.
# If MPS bfloat16 causes issues at runtime, set WAN_TRANSFORMER_DTYPE=float32 in .env.
TRANSFORMER_DTYPE = torch.bfloat16
VAE_DTYPE         = torch.float32
ENCODER_DTYPE     = torch.float32

# ─── Lazy singletons ──────────────────────────────────────────────────────────

_i2v_pipe      = None
_animate_pipe  = None


def _load_i2v():
    global _i2v_pipe
    if _i2v_pipe is not None:
        return _i2v_pipe

    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    model_id = settings.wan_i2v_model_id
    log.info(f"Loading I2V pipeline: {model_id} on {DEVICE}")
    _mem("before image_encoder load")

    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder",
        torch_dtype=ENCODER_DTYPE, low_cpu_mem_usage=True,
    )
    _mem("after image_encoder load")

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=VAE_DTYPE, low_cpu_mem_usage=True,
    )
    _mem("after vae load")

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=TRANSFORMER_DTYPE,
        low_cpu_mem_usage=True,
    )
    _mem("after transformer/pipe load")

    if settings.wan_int8:
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            log.info("Quantizing transformer to int8 (torchao)…")
            quantize_(pipe.transformer, Int8WeightOnlyConfig())
            _mem("after int8 quantization")
            log.info("int8 quantization done — transformer is now ~14 GiB (down from ~28 GiB bfloat16)")
        except Exception as exc:
            log.warning(f"int8 quantization failed ({exc}), continuing without it")

    # cpu_offload moves one sub-module at a time to MPS as it's needed, then
    # moves it back. During VAE decode only the VAE (~3 GiB) is ever on MPS,
    # so VAE decode succeeds. The full-MPS path (pipe.to(DEVICE)) puts 18+ GiB
    # on Metal simultaneously and OOMs during VAE decode on 48 GB Apple Silicon
    # despite multiple workarounds — the root cause is macOS memory pressure from
    # moving 14 GiB+ back to CPU, which puts pages in a Metal-managed state that
    # doesn't appear in RSS but still occupies physical unified memory.
    pipe.enable_model_cpu_offload(device=DEVICE)
    pipe._text_encoder_on_cpu = False
    _mem("after enable_model_cpu_offload")

    # enable_slicing only fires when batch > 1, so it does nothing for single-video inference.
    # enable_tiling tiles the spatial dims during decode, which is what actually cuts peak RAM.
    pipe.vae.enable_tiling()
    _mem("after enable_vae_tiling")

    _i2v_pipe = pipe
    log.info("I2V pipeline ready.")
    return _i2v_pipe


def _load_animate():
    global _animate_pipe
    if _animate_pipe is not None:
        return _animate_pipe

    from diffusers import AutoencoderKLWan, WanAnimatePipeline

    model_id = settings.wan_animate_model_id
    log.info(f"Loading Animate pipeline: {model_id} on {DEVICE}")

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=VAE_DTYPE, low_cpu_mem_usage=True,
    )
    pipe = WanAnimatePipeline.from_pretrained(
        model_id, vae=vae,
        torch_dtype=TRANSFORMER_DTYPE, low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload(device=DEVICE)
    _animate_pipe = pipe
    log.info("Animate pipeline ready.")
    return _animate_pipe


# ─── Image resize helper ──────────────────────────────────────────────────────

def _resize_for_wan(image: Image.Image, pipe, max_area: int) -> tuple[Image.Image, int, int]:
    """
    Resize image preserving aspect ratio so total pixels ≤ max_area.
    Dimensions must be multiples of: vae_scale_factor_spatial * patch_size[1].
    Confirmed from diffusers Wan docs.
    """
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    aspect_ratio = image.height / image.width
    height = round(math.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width  = round(math.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    height = max(mod_value, height)
    width  = max(mod_value, width)
    image  = image.resize((width, height))
    return image, int(height), int(width)


def _clamp_frames(n: int) -> int:
    """Ensure num_frames satisfies 4k+1 constraint."""
    k = max(1, (n - 1) // 4)
    return 4 * k + 1


def _decode_latents(pipe, latents: torch.Tensor):
    """
    Decode raw latents to video frames entirely on CPU.

    After denoising, Metal's private shader cache locks ~18.5 GiB of the MPS heap
    and cannot be freed with empty_cache(). Moving both transformer (~14 GiB int8)
    and VAE (~3 GiB) to CPU frees all Metal allocations, leaving ~7-8 GiB of system
    RAM for decode activations (48 GB total − 18.5 GB Metal − 14 GB transformer −
    3 GB VAE − 5 GB OS ≈ 7.5 GB headroom). Transformer and VAE are restored to MPS
    afterward for the next call.
    """
    transformer_on_mps = next(pipe.transformer.parameters()).device.type == "mps"
    if transformer_on_mps:
        log.info("Offloading transformer to CPU for VAE decode…")
        pipe.transformer.to("cpu")

    vae_on_mps = next(pipe.vae.parameters()).device.type == "mps"
    if vae_on_mps:
        log.info("Moving VAE to CPU for decode (Metal shader cache cannot be freed)…")
        pipe.vae.to("cpu")

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    _mem("after offloads + empty_cache (pre-VAE decode)")

    cfg = pipe.vae.config
    # Both VAE and latents must be on the same device (CPU).
    latents = latents.to(device="cpu", dtype=pipe.vae.dtype)
    mean    = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    std_inv = 1.0 / torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents / std_inv + mean

    try:
        _mem("before vae.decode")
        video  = pipe.vae.decode(latents, return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
        _mem("after vae.decode")
    finally:
        if vae_on_mps:
            log.info("Restoring VAE to MPS…")
            pipe.vae.to(DEVICE)
        if transformer_on_mps:
            log.info("Restoring transformer to MPS…")
            pipe.transformer.to(DEVICE)
            _mem("after transformer + VAE restore to MPS")

    return frames


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_i2v(
    selfie_path: Path,
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    num_frames: int = 81,
    num_inference_steps: Optional[int] = None,
    guidance_scale: float = 5.0,
    lora_path: Optional[Path] = None,
    lora_scale: float = 0.8,
) -> Path:
    """
    Track A / Track C: animate a selfie with a text prompt.
    Pass lora_path for Track C (style-matched output using user's trained LoRA).
    Returns path to saved MP4.
    """
    from diffusers.utils import export_to_video

    pipe   = _load_i2v()
    image  = Image.open(selfie_path).convert("RGB")
    image, height, width = _resize_for_wan(image, pipe, max_area=settings.wan_max_area)
    num_frames = _clamp_frames(num_frames)

    # Load user's style LoRA if provided (Track C).
    # Always unload after generation so next user gets a clean pipeline.
    lora_loaded = False
    if lora_path and lora_path.exists():
        log.info(f"Loading style LoRA: {lora_path}  scale={lora_scale}")
        pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[lora_scale])
        lora_loaded = True

    steps = num_inference_steps if num_inference_steps is not None else settings.wan_num_inference_steps
    log.info(f"I2V generate: {width}x{height}  {num_frames} frames  steps={steps}  lora={lora_loaded}  prompt='{prompt[:60]}'")

    # cpu_offload manages device placement per sub-module, so the pipeline handles
    # text encoding, denoising, and VAE decode internally without manual device moves.
    _mem("before pipe() call")
    try:
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height, width=width, num_frames=num_frames,
            num_inference_steps=steps, guidance_scale=guidance_scale,
        )
        frames = result.frames[0]
        _mem("after pipe() call")
    finally:
        if lora_loaded:
            pipe.unload_lora_weights()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=16)
    log.info(f"I2V saved → {output_path}")
    return output_path


def generate_animate(
    selfie_path: Path,
    pose_video_path: Path,
    face_video_path: Path,
    prompt: str,
    negative_prompt: str,
    output_path: Path,
    guidance_scale: float = 1.0,
) -> Path:
    """
    Track B: animate a selfie by replicating motion from pose+face videos
    using WanAnimatePipeline. pose_video and face_video must be preprocessed
    (run pose_service.preprocess_reference_video first).
    Returns path to saved MP4.
    """
    from diffusers.utils import export_to_video, load_video

    pipe  = _load_animate()
    image = Image.open(selfie_path).convert("RGB")
    image, height, width = _resize_for_wan(image, pipe, max_area=480 * 832)

    pose_frames = load_video(str(pose_video_path))
    face_frames = load_video(str(face_video_path))

    log.info(f"Animate generate: {width}x{height}  pose={len(pose_frames)}f  face={len(face_frames)}f")

    result = pipe(
        image=image,
        pose_video=pose_frames,
        face_video=face_frames,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        mode="animate",
    )
    frames = result.frames[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=16)
    log.info(f"Animate saved → {output_path}")
    return output_path
