"""
Replicate video generation — two modes:

  Selfie mode  (wan-2.7-i2v):
    Clip 1: first_frame=selfie
    Clip N: first_frame=last_frame(previous_clip)  (extract → feed as first_frame)

  Reference mode  (wan-2.7-r2v):
    reference_images=[url1, url2, ...]  extracted from creator's YouTube videos
    No training required — model generates in the creator's likeness directly.

Both modes: InsightFace face swap applied per clip as belt-and-suspenders.
"""
from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path
from typing import Optional

from backend.core.config import get_settings

log = logging.getLogger("replicate_service")

# ─── Schema cache + input validation ──────────────────────────────────────────

_schema_cache: dict[str, dict] = {}


def _model_schema(client, model_ref: str) -> dict:
    """Fetch and cache the Input property schema for a Replicate model."""
    if model_ref in _schema_cache:
        return _schema_cache[model_ref]
    try:
        if ":" in model_ref:
            # Versioned ref: "owner/model:version_hash" — use versions API
            model_part, version_hash = model_ref.rsplit(":", 1)
            owner, name = model_part.split("/", 1)
            version = client.models.versions.get(owner, name, version_hash)
        else:
            version = client.models.get(model_ref).latest_version
        props = (
            version.openapi_schema
            .get("components", {})
            .get("schemas", {})
            .get("Input", {})
            .get("properties", {})
        )
        # Guard: MagicMock in tests or unexpected API responses must not pass through
        if not isinstance(props, dict):
            props = {}
        _schema_cache[model_ref] = props
        log.info(f"Schema loaded for {model_ref}: {list(props.keys())}")
        return props
    except Exception as exc:
        log.warning(f"Could not fetch schema for {model_ref}: {exc} — skipping validation")
        _schema_cache[model_ref] = {}
        return {}


def _sanitize(inp: dict, schema: dict, model_ref: str) -> dict:
    """
    Remove fields not in the model's schema (prevents E006 invalid-input errors).
    Warns on dropped fields and validates enum values so issues are visible in logs.
    File-like objects are passed through without enum checking.
    """
    if not schema:
        log.warning(f"No schema for {model_ref} — sending inputs as-is: {[k for k in inp if not hasattr(inp[k],'read')]}")
        return inp

    clean: dict = {}
    for k, v in inp.items():
        if k not in schema:
            log.warning(f"[{model_ref}] dropping unknown field '{k}' (not in model schema)")
            continue

        # Validate enum constraints for scalar values
        field = schema[k]
        enums = field.get("enum") or next(
            (d.get("enum") for d in field.get("allOf", [])), None
        )
        if enums and not hasattr(v, "read"):
            if v not in enums:
                log.warning(f"[{model_ref}] field '{k}'={v!r} not in allowed values {enums}")
        clean[k] = v

    scalar_fields = {k: v for k, v in clean.items() if not hasattr(v, "read")}
    log.info(f"[{model_ref}] validated input: {scalar_fields}")
    return clean


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _client(api_key: str):
    import replicate
    return replicate.Client(api_token=api_key)


def _resolve_output(output):
    """
    Normalise client.run() output to a single downloadable object.

    The Replicate SDK may return:
      - A FileOutput object  (has .read())
      - A list of FileOutput objects
      - A generator / iterator of FileOutput objects (streaming multi-output)
    In all cases we want the *last* item (final video frame/file).
    """
    import types
    # Materialise generators / iterators; leave FileOutput / str / bytes alone
    if isinstance(output, (types.GeneratorType,)) or (
        hasattr(output, "__iter__")
        and not hasattr(output, "read")
        and not isinstance(output, (str, bytes, list))
    ):
        output = list(output)
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Replicate returned empty output list")
        return output[-1]
    return output


def _download(url, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _resolve_output(url)   # normalise in case caller skipped resolve
    if hasattr(url, "read"):
        data = url.read()
        if not data:
            raise RuntimeError(f"Replicate returned empty file for {dest.name}")
        with open(dest, "wb") as f:
            f.write(data)
    else:
        urllib.request.urlretrieve(str(url), dest)
        if dest.stat().st_size == 0:
            raise RuntimeError(f"Downloaded file is empty: {dest}")
    return dest


def _extract_last_frame(video_path: Path, out_path: Path) -> Path:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not extract last frame from {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path


# ─── Public API ───────────────────────────────────────────────────────────────

def test_connection(api_key: str) -> bool:
    try:
        client = _client(api_key)
        client.models.get("stability-ai/stable-diffusion")
        return True
    except Exception as exc:
        log.warning(f"Replicate key test failed: {exc}")
        return False


def generate_clip_i2v(
    api_key: str,
    output_path: Path,
    prompt: str,
    negative_prompt: str = "",
    first_frame_path: Optional[Path] = None,
    prev_clip_path: Optional[Path] = None,
    resolution: str = "720p",
    duration: int = 10,
) -> Path:
    """
    Generate one clip via wan-2.7-i2v (selfie / continuation mode).

    Clip 1: pass first_frame_path (selfie).
    Clip N: pass prev_clip_path — model continues naturally from that clip.
    """
    client = _client(api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    s = get_settings()
    inp: dict = {
        "prompt":       prompt,
        "resolution":   resolution,
        "duration":     min(duration, 15),
        "aspect_ratio": s.replicate_aspect_ratio,
    }

    if prev_clip_path and prev_clip_path.exists():
        last_frame = prev_clip_path.parent / f"{prev_clip_path.stem}_last.jpg"
        _extract_last_frame(prev_clip_path, last_frame)
        inp["first_frame"] = open(last_frame, "rb")
        log.info(f"i2v clip: continuing from last frame of {prev_clip_path.name}")
    elif first_frame_path and first_frame_path.exists():
        inp["first_frame"] = open(first_frame_path, "rb")
        log.info(f"i2v clip: starting from first_frame={first_frame_path.name}")
    else:
        raise ValueError("Either first_frame_path or prev_clip_path is required")

    schema = _model_schema(client, s.replicate_i2v_model)
    inp = _sanitize(inp, schema, s.replicate_i2v_model)
    output = client.run(s.replicate_i2v_model, input=inp)
    output_url = _resolve_output(output)
    log.info(f"i2v clip done → downloading")
    _download(output_url, output_path)
    return output_path


def generate_clip_r2v(
    api_key: str,
    output_path: Path,
    prompt: str,
    reference_frame_paths: list[Path],
    negative_prompt: str = "",
    resolution: str = "720p",
    duration: int = 10,
) -> Path:
    """
    Generate one clip via wan-2.7-r2v (YouTube reference mode).

    reference_frame_paths: curated face frames extracted from YouTube.
    No first_frame or training needed — model generates in the person's likeness.
    """
    if not reference_frame_paths:
        raise ValueError("At least one reference frame is required for r2v mode")

    s = get_settings()
    client = _client(api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_files = [open(p, "rb") for p in reference_frame_paths if p.exists()][:5]  # model limit
    if not ref_files:
        raise ValueError("No valid reference frame files found")

    inp: dict = {
        "prompt":           prompt,
        "reference_images": ref_files,
        "resolution":       resolution,
        "duration":         min(duration, 10),
        "aspect_ratio":     s.replicate_aspect_ratio,
    }

    schema = _model_schema(client, s.replicate_r2v_model)
    inp = _sanitize(inp, schema, s.replicate_r2v_model)
    log.info(f"r2v clip: {len(ref_files)} reference frames  prompt='{prompt[:60]}'")
    output = client.run(s.replicate_r2v_model, input=inp)
    output_url = _resolve_output(output)
    log.info(f"r2v clip done → downloading")
    _download(output_url, output_path)

    for f in ref_files:
        f.close()
    return output_path


def generate_full_video(
    api_key: str,
    selfie_path: Path,
    prompt: str,
    output_path: Path,
    reference_frame_paths: Optional[list[Path]] = None,
    lora_model_ref: Optional[str] = None,
    lora_version_id: Optional[str] = None,
    lora_trigger_word: Optional[str] = None,
    lora_weights_url: Optional[str] = None,
    num_clips: int | None = None,
    resolution: str = "720p",
    on_progress=None,
) -> Path:
    """
    Generate a full video. Priority order:
      1. LoRA model (best identity)  — lora_model_ref + lora_version_id provided
      2. r2v reference mode           — reference_frame_paths provided (no face swap)
      3. i2v selfie mode              — fallback (face swap applied)
    """
    from backend.services import face_swap_service, video_chain_service

    s = get_settings()
    use_lora      = bool(lora_model_ref and lora_version_id)
    use_reference = bool(reference_frame_paths) and not use_lora
    n = num_clips if num_clips is not None else s.replicate_clips_per_video
    duration = s.replicate_clip_duration
    tmp_dir = Path("/tmp/stylevid2/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    mode = "lora" if use_lora else ("r2v" if use_reference else "i2v")
    log.info(f"generate_full_video: mode={mode}  clips={n}  resolution={resolution}")

    raw_clips: list[Path] = []
    swapped_clips: list[Path] = []

    total_steps = n * 2 + 1
    step = 0

    def _pct(s: int) -> int:
        return int(s / total_steps * 95)

    for i in range(n):
        label = f"clip_{i+1}_of_{n}"

        # ── Generate ──────────────────────────────────────────────────────────
        step += 1
        if on_progress:
            on_progress(_pct(step), f"Generating clip {i+1}/{n} on Replicate…")

        raw_path = tmp_dir / f"{label}_raw.mp4"

        if use_lora:
            generate_clip_lora(
                api_key=api_key,
                output_path=raw_path,
                prompt=prompt,
                lora_model_ref=lora_model_ref,
                lora_version_id=lora_version_id,
                trigger_word=lora_trigger_word or "",
                lora_weights_url=lora_weights_url,
            )
        elif use_reference:
            generate_clip_r2v(
                api_key=api_key,
                output_path=raw_path,
                prompt=prompt,
                reference_frame_paths=reference_frame_paths,
                resolution=resolution,
                duration=duration,
            )
        else:
            prev_clip = swapped_clips[-1] if swapped_clips else None
            generate_clip_i2v(
                api_key=api_key,
                output_path=raw_path,
                prompt=prompt,
                first_frame_path=selfie_path if not prev_clip else None,
                prev_clip_path=prev_clip,
                resolution=resolution,
                duration=duration,
            )

        raw_clips.append(raw_path)

        # ── Face swap ─────────────────────────────────────────────────────────
        step += 1
        if on_progress:
            on_progress(_pct(step), f"Processing clip {i+1}/{n}…")

        if use_reference:
            # r2v generates identity natively from reference frames; face swap degrades quality
            swapped_clips.append(raw_path)
        else:
            # Apply face swap for both i2v and LoRA modes:
            # - i2v: primary identity enforcement
            # - LoRA: belt-and-suspenders in case the LoRA didn't fully lock the face
            swapped_path = tmp_dir / f"{label}_swapped.mp4"
            try:
                face_swap_service.swap_video(
                    video_path=raw_path,
                    source_face_path=selfie_path,
                    output_path=swapped_path,
                )
                swapped_clips.append(swapped_path)
            except Exception as exc:
                log.warning(f"Face swap failed for clip {i+1} ({exc}) — using raw clip")
                swapped_clips.append(raw_path)

    # ── Stitch ────────────────────────────────────────────────────────────────
    if on_progress:
        on_progress(97, "Stitching clips…")

    video_chain_service.stitch(swapped_clips, output_path)

    if on_progress:
        on_progress(100, "Done!")

    log.info(f"Full video ready → {output_path}")
    return output_path


# ─── LoRA training helpers ────────────────────────────────────────────────────

_LORA_TRAINER = "zsxkib/hunyuan-video-lora"
_LORA_INFERENCE_PARAMS = {
    "num_frames":            65,   # ~2.7s @ 24fps
    "num_inference_steps":   30,
    "guidance_scale":        6.0,
    "flow_shift":            7.0,
    "width":                 544,
    "height":                960,
}


def get_replicate_username(api_key: str) -> str:
    """Return the Replicate account username for this API key."""
    import requests
    r = requests.get(
        "https://api.replicate.com/v1/account",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["username"]


# Hardware preference order for LoRA destination models.
# Replicate accepts a single SKU, so we pick the best available from this list.
_LORA_HARDWARE_PREFERENCE = [
    "gpu-h100",
    "gpu-a100-large-2x",
    "gpu-a100-large",
    "gpu-l40s-2x",
    "gpu-l40s",
    "gpu-t4",
    "cpu",
]


def _extract_allowed_hardware(resp) -> list[str]:
    """Parse allowed hardware SKUs from Replicate validation responses."""
    try:
        body = resp.json()
    except Exception:
        body = {}

    candidates: list[str] = []

    # Preferred parse path: structured error entries.
    for err in body.get("errors", []) if isinstance(body, dict) else []:
        detail = err.get("detail") if isinstance(err, dict) else ""
        if isinstance(detail, str) and "Your options are:" in detail:
            options = detail.split("Your options are:", 1)[1]
            candidates.extend([x.strip().strip(".") for x in options.split(",") if x.strip()])

    # Fallback parse path: free-form detail text.
    detail_text = body.get("detail") if isinstance(body, dict) else ""
    if isinstance(detail_text, str) and "Your options are:" in detail_text:
        options = detail_text.split("Your options are:", 1)[1]
        candidates.extend([x.strip().strip(".") for x in options.split(",") if x.strip()])

    # Last fallback: regex against raw body text.
    if not candidates:
        text = getattr(resp, "text", "") or ""
        m = re.search(r"Your options are:\s*([^\n]+)", text)
        if m:
            candidates.extend([x.strip().strip(".") for x in m.group(1).split(",") if x.strip()])

    # Stable de-dupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for sku in candidates:
        if sku and sku not in seen:
            seen.add(sku)
            out.append(sku)
    return out


def _select_best_hardware(allowed: list[str]) -> str | None:
    """Choose the highest-priority hardware SKU available for this account."""
    if not allowed:
        return None
    allowed_set = set(allowed)
    for sku in _LORA_HARDWARE_PREFERENCE:
        if sku in allowed_set:
            return sku
    return allowed[0]


def ensure_destination_model(api_key: str, username: str, model_name: str) -> str:
    """
    Create (or fix) a private destination model in the user's Replicate account.
    If the model exists with cpu hardware, patches it to GPU so trained versions
    can serve predictions. Returns the model ref string.
    """
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model_ref = f"{username}/{model_name}"

    get_r = requests.get(
        f"https://api.replicate.com/v1/models/{model_ref}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if get_r.status_code == 200:
        existing = get_r.json()
        current_hw = existing.get("default_example", {}) or {}
        current_hw_value = existing.get("hardware") or current_hw.get("hardware")
        # PATCH to upgrade hardware if it was created as cpu
        if current_hw_value == "cpu":
            preferred = _select_best_hardware(_LORA_HARDWARE_PREFERENCE) or "gpu-a100-large"
            patch_r = requests.patch(
                f"https://api.replicate.com/v1/models/{model_ref}",
                headers=headers,
                json={"hardware": preferred},
                timeout=15,
            )
            if patch_r.status_code == 200:
                log.info(f"Upgraded destination model hardware to {preferred}: {model_ref}")
            elif patch_r.status_code == 400:
                allowed = _extract_allowed_hardware(patch_r)
                fallback = _select_best_hardware(allowed)
                if fallback and fallback != preferred:
                    retry_r = requests.patch(
                        f"https://api.replicate.com/v1/models/{model_ref}",
                        headers=headers,
                        json={"hardware": fallback},
                        timeout=15,
                    )
                    if retry_r.status_code == 200:
                        log.info(f"Upgraded destination model hardware to {fallback}: {model_ref}")
                    else:
                        log.warning(
                            f"Could not patch hardware on {model_ref}: {retry_r.status_code} {retry_r.text}"
                        )
                else:
                    log.warning(f"Could not patch hardware on {model_ref}: {patch_r.status_code} {patch_r.text}")
            else:
                log.warning(f"Could not patch hardware on {model_ref}: {patch_r.status_code} {patch_r.text}")
        else:
            log.info(f"Destination model already exists: {model_ref}")
        return model_ref

    first_choice = _select_best_hardware(_LORA_HARDWARE_PREFERENCE) or "gpu-a100-large"
    payload = {
        "owner": username,
        "name": model_name,
        "visibility": "private",
        "hardware": first_choice,
        "description": "StyleVid face LoRA — auto-created",
    }
    r = requests.post("https://api.replicate.com/v1/models", headers=headers, json=payload, timeout=30)

    # If hardware is invalid for this account/token, parse allowed options and retry once.
    if r.status_code == 400:
        allowed = _extract_allowed_hardware(r)
        fallback = _select_best_hardware(allowed)
        if fallback and fallback != first_choice:
            log.info(
                f"Replicate account doesn't allow '{first_choice}', retrying model create with '{fallback}'"
            )
            payload["hardware"] = fallback
            r = requests.post("https://api.replicate.com/v1/models", headers=headers, json=payload, timeout=30)

    if r.status_code in (200, 201):
        created_hw = payload.get("hardware")
        log.info(f"Destination model created ({created_hw}): {model_ref}")
    elif r.status_code == 422:
        log.info(f"Destination model already exists (422): {model_ref}")
    else:
        log.error(f"Failed to create destination model {model_ref}: HTTP {r.status_code} — {r.text}")
        raise RuntimeError(
            "Could not create your Replicate model right now. "
            "Please verify your Replicate API key and try again."
        )
    return model_ref


def start_lora_training(
    api_key: str,
    destination_model_ref: str,
    zip_path: Path,
    trigger_word: str,
    steps: int = 1500,
) -> str:
    """
    Submit a HunyuanVideo LoRA training job on Replicate.
    Returns the Replicate training ID.
    steps: training iterations — more gives stronger identity but takes longer.
           1500 is a reliable default; 1000 is minimum viable; 2000 is high quality.
    """
    client = _client(api_key)
    trainer = client.models.get(_LORA_TRAINER)
    trainer_version = trainer.latest_version.id

    log.info(
        f"Starting LoRA training: trainer={_LORA_TRAINER}:{trainer_version[:8]}  "
        f"destination={destination_model_ref}  trigger={trigger_word}  steps={steps}"
    )

    training = client.trainings.create(
        version=f"{_LORA_TRAINER}:{trainer_version}",
        input={
            "input_videos":  open(zip_path, "rb"),
            "trigger_word":  trigger_word,
            "autocaption":   True,
            "steps":         steps,
        },
        destination=destination_model_ref,
    )
    log.info(f"LoRA training started: id={training.id}  steps={steps}")
    return training.id


def poll_lora_training(api_key: str, training_id: str) -> dict:
    """
    Return current status of a Replicate training.
    Dict keys: status, version_id, weights_url, error.
    weights_url is the direct URL to the .safetensors file — used for Option B inference.
    """
    client = _client(api_key)
    training = client.trainings.get(training_id)
    version_id = None
    weights_url = None
    if training.status == "succeeded" and training.output:
        version_id  = training.output.get("version")
        weights_url = training.output.get("weights")
        if not weights_url:
            # Some trainer versions use a different key
            weights_url = training.output.get("lora_weights") or training.output.get("output_weights")
    return {
        "status":      training.status,
        "version_id":  version_id,
        "weights_url": weights_url,
        "error":       training.error,
    }


def generate_clip_lora(
    api_key: str,
    output_path: Path,
    prompt: str,
    lora_model_ref: str,
    lora_version_id: str,
    trigger_word: str = "",
    lora_weights_url: str | None = None,
) -> Path:
    """
    Generate one clip using the user's trained HunyuanVideo LoRA model.

    Inference strategy (in priority order):
      Option B (primary):  pass lora_weights URL to base zsxkib/hunyuan-video-lora model.
                           Avoids destination model hardware requirements entirely.
      Option A (fallback): run the versioned destination model directly.
                           Requires the model to have GPU hardware configured.
    """
    client = _client(api_key)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_prompt = f"{trigger_word}, {prompt}" if trigger_word else prompt
    log.info(f"lora clip: model={lora_model_ref}  weights={'url' if lora_weights_url else 'version'}  prompt='{full_prompt[:80]}'")

    base_inp = {**_LORA_INFERENCE_PARAMS, "prompt": full_prompt}

    # Always use Option A (versioned destination model with GPU hardware).
    # Option B (weights URL to trainer) disabled — zsxkib/hunyuan-video-lora is trainer-only,
    # not an inference model. Using it with inference params produces empty/garbage output.
    # Future: investigate if there's a separate HunyuanVideo base inference model that accepts lora_weights.
    if lora_weights_url:
        log.info(f"lora inference: weights_url provided but using Option A (destination model)")
    
    # Option A: destination model version directly (GPU-capable)
    versioned_ref = lora_version_id if ":" in lora_version_id else f"{lora_model_ref}:{lora_version_id}"
    ref = versioned_ref
    inp = base_inp
    schema = _model_schema(client, versioned_ref)
    inp = _sanitize(inp, schema, versioned_ref)
    log.info(f"lora inference via destination model: {versioned_ref}")

    output = client.run(ref, input=inp)
    output_url = _resolve_output(output)
    log.info("lora clip done → downloading")
    _download(output_url, output_path)
    return output_path
