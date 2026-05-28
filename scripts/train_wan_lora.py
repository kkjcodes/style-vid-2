"""
Train a style LoRA on Wan2.1-I2V-14B using the user's own video clips.

Two-phase to minimize peak RAM:
  Phase 1 — precompute: load VAE, CLIP, UMT5 one at a time → encode all
             clips and embeddings → save to disk → unload each encoder.
  Phase 2 — train:      load transformer + LoRA (CPU, float32) → flow
             matching training loop → save LoRA weights.

Flow matching (Wan2.1 uses FM, not DDPM):
  t ∈ [0, 1),  noisy = (1-t)*latent + t*noise,  target = noise - latent

Wan I2V 36-channel input:
  cat([noisy_16ch, zeros_4ch, first_frame_latent_16ch_expanded], dim=1)

LoRA save format — diffusers-compatible:
  Strip "base_model.model." prefix and ".default" adapter name from
  PEFT keys so pipe.load_lora_weights() works without patching.

Usage:
  PYTHONPATH=. python scripts/train_wan_lora.py \\
    --video_dir /tmp/stylevid/youtube_downloads/<user_id> \\
    --output_dir /tmp/stylevid2/loras/<user_id> \\
    --trigger_word sks \\
    --num_steps 500 \\
    --lora_rank 16

  # Skip precompute if you already have the cache:
  PYTHONPATH=. python scripts/train_wan_lora.py \\
    --video_dir ... --output_dir ... --skip_precompute
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("train_wan_lora")


# ─── Video loading ────────────────────────────────────────────────────────────

def load_video_frames(
    video_path: Path, num_frames: int, height: int, width: int
) -> list:
    """Return exactly num_frames evenly-spaced PIL frames, resized to (width, height)."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, num_frames)

    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            frames.append(frames[-1] if frames else Image.new("RGB", (width, height)))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb).resize((width, height)))
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def frames_to_tensor(frames: list) -> torch.Tensor:
    """Convert list of PIL frames → [1, 3, T, H, W] float32 in [-1, 1]."""
    arr = np.stack([np.array(f, dtype=np.float32) for f in frames])  # [T, H, W, 3]
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # [1, 3, T, H, W]


# ─── Phase 1: Precompute ──────────────────────────────────────────────────────

def precompute(
    video_dir: Path,
    model_id: str,
    cache_dir: Path,
    num_frames: int,
    height: int,
    width: int,
    trigger_word: str,
) -> None:
    from diffusers import AutoencoderKLWan
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
        T5TokenizerFast,
        UMT5EncoderModel,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    video_paths = sorted(
        list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4"))
    )
    if not video_paths:
        raise ValueError(f"No .mp4 files found in {video_dir}")
    log.info(f"Found {len(video_paths)} video(s) in {video_dir}")

    # ── Text embeddings (shared across all clips) ────────────────────────────
    text_emb_path = cache_dir / "text_embeds.pt"
    if text_emb_path.exists():
        log.info("Text embeddings already cached — skipping.")
    else:
        prompt = f"a high quality video of {trigger_word} person, natural movement"
        log.info(f"Encoding prompt with UMT5: '{prompt}'")

        tokenizer = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder",
            torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
        text_encoder.eval()

        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            # attention_mask required: without it the encoder attends to padding tokens
            text_embeds = text_encoder(
                tokens.input_ids, attention_mask=tokens.attention_mask
            ).last_hidden_state  # [1, 512, dim]

        torch.save(text_embeds, text_emb_path)
        log.info(f"  text_embeds shape: {tuple(text_embeds.shape)}  → saved.")

        del text_encoder, tokenizer
        gc.collect()

    # ── VAE + CLIP (encode all clips together before unloading) ──────────────
    log.info("Loading VAE (float32)…")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    vae.eval()

    log.info("Loading CLIP image encoder (float32)…")
    clip_processor = CLIPImageProcessor.from_pretrained(
        model_id, subfolder="image_processor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        model_id, subfolder="image_encoder",
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    image_encoder.eval()

    encoded_count = 0
    for i, vp in enumerate(video_paths):
        latent_path = cache_dir / f"latent_{i:04d}.pt"
        img_emb_path = cache_dir / f"img_emb_{i:04d}.pt"

        if latent_path.exists() and img_emb_path.exists():
            log.info(f"[{i+1}/{len(video_paths)}] {vp.name} — cached, skipping.")
            encoded_count += 1
            continue

        log.info(f"[{i+1}/{len(video_paths)}] Encoding {vp.name} …")
        frames = load_video_frames(vp, num_frames, height, width)

        # VAE encode: [1, 3, T, H, W] → [1, 16, T', H', W']
        video_tensor = frames_to_tensor(frames)
        with torch.no_grad():
            posterior = vae.encode(video_tensor).latent_dist
            latent = posterior.sample()

        # Wan per-channel normalization — config values are Python lists, reshape to [1,16,1,1,1]
        mean = torch.tensor(vae.config.latents_mean, dtype=latent.dtype).view(1, -1, 1, 1, 1)
        std  = torch.tensor(vae.config.latents_std,  dtype=latent.dtype).view(1, -1, 1, 1, 1)
        latent = (latent - mean) / std

        torch.save(latent, latent_path)

        # CLIP encode first frame → hidden_states[-2]: [1, 257, 1280]
        pv = clip_processor(images=frames[0], return_tensors="pt").pixel_values
        with torch.no_grad():
            out = image_encoder(pv, output_hidden_states=True)
            img_emb = out.hidden_states[-2]  # second-to-last layer, confirmed from Wan I2V pipeline

        torch.save(img_emb, img_emb_path)

        log.info(f"  latent={tuple(latent.shape)}  img_emb={tuple(img_emb.shape)}")
        encoded_count += 1

    del vae, image_encoder, clip_processor
    gc.collect()
    log.info(f"Precompute done — {encoded_count} clips cached in {cache_dir}")


# ─── Phase 2: Train ───────────────────────────────────────────────────────────

def _quantize_frozen_layers(peft_model: torch.nn.Module) -> bool:
    """
    Int8 weight-only quantization on all frozen Linear layers, called after PEFT wrapping.

    Why after PEFT: if we quantize first, PEFT detects AffineQuantizedTensor and
    dispatches through TorchaoLoraLinear, whose API changed in torchao 0.17 (missing
    get_apply_tensor_subclass). Quantizing after PEFT wrapping avoids that path —
    PEFT already holds plain nn.Linear references via base_layer, and quantize_()
    converts those in-place transparently.

    Memory impact: bfloat16 weights (2 bytes) → int8 (1 byte) = 28GB → ~14GB.
    This lets us drop gradient checkpointing entirely, which was recomputing all
    40 transformer blocks on every backward pass (~3x compute overhead).
    With int8: 14GB weights + ~12-16GB activations ≈ 26-30GB < 47.74GB MPS limit.

    filter_fn: only quantize frozen layers (requires_grad=False), so LoRA A/B
    matrices remain in bfloat16 with full gradients.
    """
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
    except ImportError:
        log.warning(
            "torchao not found — skipping int8 quantization. "
            "Training will still work but will be slower (gradient checkpointing will be used). "
            "Install with: pip install torchao"
        )
        return False

    def _is_frozen_linear(module: torch.nn.Module, fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear) and not module.weight.requires_grad

    quantize_(peft_model, Int8WeightOnlyConfig(), filter_fn=_is_frozen_linear)
    log.info("int8 weight quantization applied — frozen layers ~14GB (was 28GB bfloat16)")
    return True


def _save_lora(peft_model, output_dir: Path, lora_rank: int, lora_alpha: float) -> None:
    """
    Save LoRA weights in diffusers-compatible format.

    PEFT 0.19 saves:   base_model.model.layers.X.attn1.to_q.lora_A.default.weight
    diffusers expects:  transformer.layers.X.attn1.to_q.lora_A.weight

    Transformation: strip "base_model.model.", add "transformer.", drop ".default".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_sd = peft_model.state_dict()

    diffusers_sd = {}
    for k, v in raw_sd.items():
        if "lora_A" not in k and "lora_B" not in k:
            continue
        new_k = k.replace("base_model.model.", "transformer.")
        new_k = new_k.replace(".default.", ".")
        diffusers_sd[new_k] = v

    if not diffusers_sd:
        log.error("No LoRA keys found in state_dict — something is wrong with PEFT wrapping.")
        return

    meta = {
        "lora_rank": str(lora_rank),
        "lora_alpha": str(lora_alpha),
        "base_model": "Wan2.1-I2V-14B",
        "target_modules": "to_q,to_k,to_v,to_out.0",
    }
    try:
        from safetensors.torch import save_file
        save_file(diffusers_sd, str(output_dir / "lora_weights.safetensors"), metadata=meta)
        log.info(f"  {len(diffusers_sd)} tensors → {output_dir}/lora_weights.safetensors")
    except ImportError:
        torch.save({"weights": diffusers_sd, "meta": meta}, output_dir / "lora_weights.pt")
        log.info(f"  {len(diffusers_sd)} tensors → {output_dir}/lora_weights.pt")


def train(
    cache_dir: Path,
    model_id: str,
    output_dir: Path,
    num_steps: int,
    lr: float,
    lora_rank: int,
    lora_alpha: float,
) -> None:
    import torch.nn.functional as F
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig, get_peft_model

    latent_paths  = sorted(cache_dir.glob("latent_*.pt"))
    img_emb_paths = sorted(cache_dir.glob("img_emb_*.pt"))
    text_emb_path = cache_dir / "text_embeds.pt"

    if not latent_paths:
        raise ValueError(f"No precomputed latents found in {cache_dir}. Run precompute first.")
    if len(latent_paths) != len(img_emb_paths):
        raise ValueError("Mismatch between latent and image embedding counts in cache.")

    # Device: MPS if available (M-series GPU), else CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # bfloat16 on MPS (native M-series support, half the RAM of float32)
    # float32 on CPU (MPS bfloat16 backprop is safe for LoRA since only adapter params have gradients)
    dtype = torch.bfloat16 if device.type == "mps" else torch.float32
    log.info(f"Training device: {device}  dtype: {dtype}")

    text_embeds = torch.load(text_emb_path, weights_only=True).to(device=device, dtype=dtype)
    n_clips = len(latent_paths)
    log.info(f"Training on {n_clips} clip(s) for {num_steps} steps.")

    # Load transformer on CPU first (low_cpu_mem_usage streams weights, halves peak RAM)
    log.info(f"Loading transformer ({model_id}) — dtype={dtype} …")
    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer",
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    # Step 1: Apply LoRA on CPU — PEFT must see plain nn.Linear before any quantization.
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
        ],
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Step 2: Quantize frozen base layers to int8 — after PEFT so PEFT never sees
    # AffineQuantizedTensor (avoids the TorchaoLoraLinear dispatch path).
    # Returns True if quantization succeeded; falls back to gradient checkpointing if not.
    quantized = _quantize_frozen_layers(transformer)

    # Step 3: Move to MPS — int8 tensors transfer cleanly to MPS unified memory.
    transformer = transformer.to(device)

    # Gradient checkpointing only as a fallback when int8 quantization is unavailable.
    # With int8 weights (~14GB), peak memory is ~26-30GB, safely under 47.74GB MPS limit.
    # With bfloat16 weights (~28GB), GC is required to avoid OOM, but costs ~3x compute.
    if not quantized:
        log.warning(
            "Falling back to gradient checkpointing — install torchao for 3x faster training."
        )
        transformer.base_model.model.enable_gradient_checkpointing()

    # LoRA params only — base model is frozen (requires_grad=False), no gradients stored for it.
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    # foreach=False: MPS does not support all foreach (vectorized) optimizer ops
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4, foreach=False)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Preload all latents and image embeddings into RAM — eliminates per-step disk I/O.
    # 25 clips × ~1.3MB each = ~33MB total, trivially fits in memory.
    log.info("Preloading latents and image embeddings into RAM…")
    all_latents = [
        torch.load(p, weights_only=True).to(device=device, dtype=dtype) for p in latent_paths
    ]
    all_img_embs = [
        torch.load(p, weights_only=True).to(device=device, dtype=dtype) for p in img_emb_paths
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    transformer.train()
    recent_losses: list[float] = []

    for step in range(1, num_steps + 1):
        clip_idx = (step - 1) % n_clips

        latent  = all_latents[clip_idx]
        img_emb = all_img_embs[clip_idx]

        # Slice to first 3 latent frames (9 pixel frames equivalent).
        # Precomputed latents have 5 frames (17 pixel frames); slicing to 3 halves
        # the attention sequence length (3200 → 1920 tokens), cutting peak memory by ~60%.
        # Style is fully encoded in any 3-frame window — no quality loss for LoRA training.
        latent = latent[:, :, :3, :, :]

        B, C, T_lat, H_lat, W_lat = latent.shape

        # Flow matching timestep: integer in [0, 1000), continuous t = timestep/1000
        t_int = torch.randint(0, 1000, (B,), device=device)
        t_bc  = (t_int.float() / 1000.0).to(dtype).view(B, 1, 1, 1, 1)

        noise        = torch.randn_like(latent)
        noisy_latent = (1.0 - t_bc) * latent + t_bc * noise
        vel_target   = noise - latent  # flow matching velocity field

        # I2V condition: first frame latent expanded as spatial reference.
        # mask_4ch: 1 at frame 0 (conditioned), 0 elsewhere (to generate).
        # Mirrors the pipeline's prepare_latents: mask[:,:,0,:,:]=1 after reshape
        # from pixel-space [B,1,T_pix,H,W] → latent-space [B,4,T_lat,H,W].
        first_frame = latent[:, :, :1, :, :].expand(-1, -1, T_lat, -1, -1)
        mask_4ch    = torch.zeros(B, 4, T_lat, H_lat, W_lat, device=device, dtype=dtype)
        mask_4ch[:, :, 0, :, :] = 1.0  # first latent frame is the condition

        # 36-channel transformer input: [noisy_16 | mask_4 | first_frame_16]
        hidden_states = torch.cat([noisy_latent, mask_4ch, first_frame], dim=1)

        pred = transformer(
            hidden_states=hidden_states,
            timestep=t_int,
            encoder_hidden_states=text_embeds.expand(B, -1, -1),
            encoder_hidden_states_image=img_emb.expand(B, -1, -1),
        )

        # Cast to float32 for loss — bfloat16 MSE can be numerically unstable
        loss = F.mse_loss(pred.sample.float(), vel_target.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        optimizer.step()
        lr_sched.step()

        recent_losses.append(loss.item())
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        if step % 50 == 0 or step == 1:
            avg = sum(recent_losses) / len(recent_losses)
            log.info(
                f"Step {step:>{len(str(num_steps))}}/{num_steps}  "
                f"loss={avg:.4f}  lr={lr_sched.get_last_lr()[0]:.2e}"
            )

        # Checkpoint every 200 steps
        if step % 200 == 0 and step < num_steps:
            ckpt = output_dir / f"checkpoint_{step}"
            log.info(f"Saving checkpoint → {ckpt}")
            _save_lora(transformer, ckpt, lora_rank, lora_alpha)

    log.info("Training complete. Saving final LoRA weights…")
    _save_lora(transformer, output_dir, lora_rank, lora_alpha)
    log.info(f"Done. Load at inference with: pipe.load_lora_weights('{output_dir}')")


def save_trigger_word(output_dir: Path, trigger_word: str) -> None:
    (output_dir / "trigger_word.txt").write_text(trigger_word)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train style LoRA for Wan2.1-I2V-14B on user's video clips."
    )
    parser.add_argument("--video_dir",  required=True, type=Path,
                        help="Directory containing user's .mp4 training clips")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Where to save final LoRA weights")
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                        help="HuggingFace model ID for Wan2.1-I2V")
    parser.add_argument("--trigger_word", default="sks",
                        help="Token embedded in training prompt (include at inference for style)")
    parser.add_argument("--num_frames", type=int, default=17,
                        help="Frames per training clip (must satisfy 4k+1; 17=4*4+1)")
    parser.add_argument("--height",    type=int, default=320, help="Training height in pixels")
    parser.add_argument("--width",     type=int, default=512, help="Training width in pixels")
    parser.add_argument("--num_steps", type=int, default=500, help="Gradient update steps")
    parser.add_argument("--lr",        type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank r")
    parser.add_argument("--lora_alpha",type=float, default=32.0, help="LoRA alpha (scale = alpha/r)")
    parser.add_argument("--skip_precompute", action="store_true",
                        help="Skip Phase 1 if cache already exists")
    args = parser.parse_args()

    # Enforce 4k+1 frame count — Wan VAE temporal compression requires this
    if (args.num_frames - 1) % 4 != 0:
        corrected = max(9, ((args.num_frames - 1) // 4) * 4 + 1)
        log.warning(f"num_frames={args.num_frames} violates 4k+1 constraint → using {corrected}")
        args.num_frames = corrected

    cache_dir = args.output_dir / "_cache"

    if not args.skip_precompute:
        precompute(
            video_dir=args.video_dir,
            model_id=args.model_id,
            cache_dir=cache_dir,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            trigger_word=args.trigger_word,
        )

    train(
        cache_dir=cache_dir,
        model_id=args.model_id,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # Save trigger word so the inference pipeline can inject it automatically
    save_trigger_word(args.output_dir, args.trigger_word)
    log.info(f"Trigger word '{args.trigger_word}' saved to {args.output_dir}/trigger_word.txt")


if __name__ == "__main__":
    main()
