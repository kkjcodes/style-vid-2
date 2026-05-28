"""
Celery tasks for YouTube-based face data preparation:

  extract_reference_frames  — still-frame extraction for wan-2.7-r2v (fast, free)
  run_lora_training         — video-clip extraction + Replicate LoRA training (~$5-10)

LoRA flow:
  1. yt-dlp downloads videos
  2. Scan for face-containing segments (InsightFace, score ≥ 0.7, face ≥ 100 px)
  3. Extract 3-5 s clips with ffmpeg (target: 15-25 clips)
  4. Zip clips → upload to Replicate training API
  5. Poll until training completes (≤ 45 min), store version_id in DB
"""
from __future__ import annotations

import json
import logging
import subprocess
import shutil
import time
import zipfile
from pathlib import Path

from backend.workers.celery_app import celery_app
from backend.core.config import get_settings
from backend.core.logging_config import setup_logging
from backend.core.security import decrypt_key

setup_logging()
log = logging.getLogger("training_worker")
settings = get_settings()

# Clip quality thresholds
_MIN_FACE_SCORE = 0.70
_MIN_FACE_PX    = 100   # minimum face bounding-box side in pixels
_CLIP_DURATION  = 4     # seconds per clip
_MIN_CLIPS      = 15    # below this the model won't reliably learn the face
_MAX_CLIPS      = 25
_SCAN_INTERVAL  = 2.0   # scan one frame every N seconds

# Training hyperparameters
_LORA_TRAIN_STEPS = 1500  # default training steps; more = better identity, longer runtime


@celery_app.task(
    bind=True,
    name="backend.workers.training_worker.extract_reference_frames",
    soft_time_limit=1800,
    time_limit=1900,
)
def extract_reference_frames(
    self,
    user_id: str,
    youtube_urls: list[str] = [],
    max_frames: int | None = None,
):
    """
    Download YouTube videos, extract the best face frames, delete the videos.
    Saves frames to storage and returns the list of frame paths.
    """
    n_frames = max_frames or settings.max_reference_frames
    tmp_dir = Path("/tmp/stylevid2/tmp") / f"yt_{user_id}"
    frames_dir = Path(settings.local_storage_dir) / "users" / user_id / "reference_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    self.update_state(state="PROGRESS", meta={"stage": "downloading", "progress_pct": 5})
    log.info(f"[{user_id}] Extracting reference frames from {len(youtube_urls)} URL(s)")

    # ── Download YouTube videos (if any URLs provided) ────────────────────────
    downloaded: list[Path] = []
    failed_urls: list[str] = []
    for url in youtube_urls:
        video_path = _download_video(url, tmp_dir)
        if video_path:
            downloaded.append(video_path)
        else:
            failed_urls.append(url)
    if failed_urls:
        log.warning(f"[{user_id}] Failed to download: {failed_urls}")

    # ── Include locally uploaded videos ──────────────────────────────────────
    from backend.services import storage_service as _storage
    for local_vid in _storage.training_video_paths(user_id):
        downloaded.append(local_vid)
        log.info(f"[{user_id}] Using uploaded video: {local_vid.name}")

    if not downloaded:
        raise RuntimeError(
            "No videos available. Upload video files using the 'Boost Quality' section."
        )

    # Normalize AV1 videos to H.264 for reliable OpenCV/ffmpeg processing.
    downloaded = [_normalize_video_for_processing(v, tmp_dir, user_id) for v in downloaded]

    log.info(f"[{user_id}] Processing {len(downloaded)} video(s)")
    self.update_state(state="PROGRESS", meta={"stage": "extracting", "progress_pct": 30})

    # ── Extract candidate frames ───────────────────────────────────────────────
    candidates: list[tuple[float, Path]] = []  # (score, frame_path)
    candidate_dir = tmp_dir / "candidates"
    candidate_dir.mkdir(exist_ok=True)

    for video_path in downloaded:
        frames = _sample_frames(video_path, candidate_dir, sample_every_n_sec=2)
        scored = _score_faces(frames)
        candidates.extend(scored)

    self.update_state(state="PROGRESS", meta={"stage": "selecting", "progress_pct": 70})

    if not candidates:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            "No usable face frames found. "
            "Ensure the YouTube video clearly shows the creator's face."
        )

    # ── Check LoRA viability early (same videos, same logic as run_lora_training) ──
    # Count qualifying clips now so the user sees the "need more videos" error
    # at the Extract Frames step, not later at the Train Likeness step.
    self.update_state(state="PROGRESS", meta={"stage": "checking training readiness…", "progress_pct": 72})
    clips_check_dir = tmp_dir / "clips_check"
    clips_check_dir.mkdir(exist_ok=True)
    lora_clip_count = 0
    for vid in downloaded:
        clips = _segment_face_clips(vid, clips_check_dir)
        lora_clip_count += len(clips)
        for c in clips:
            c.unlink(missing_ok=True)  # count only — delete immediately
    clips_check_dir.rmdir()

    if lora_clip_count < _MIN_CLIPS:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"Only {lora_clip_count} usable face clips found — need at least {_MIN_CLIPS} "
            f"for reliable identity learning. Please add more YouTube URLs "
            f"(aim for 3–5 videos showing your face in varied settings and angles)."
        )

    log.info(f"[{user_id}] LoRA viability check passed: {lora_clip_count} clips available")

    # ── Deduplicate then select top N frames ─────────────────────────────────
    # Sort by score descending, then deduplicate by MD5 (same frame extracted
    # from consecutive timestamps scores identically and inflates top-N).
    import hashlib
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen_hashes: set[str] = set()
    unique_candidates: list[tuple[float, Path]] = []
    for score, path in candidates:
        h = hashlib.md5(path.read_bytes()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_candidates.append((score, path))

    selected = unique_candidates[:n_frames]
    log.info(
        f"[{user_id}] {len(candidates)} face frames found, "
        f"{len(unique_candidates)} unique, keeping top {len(selected)}"
    )

    # Clear previous reference frames for this user, save new ones
    for old in frames_dir.glob("frame_*.jpg"):
        old.unlink(missing_ok=True)

    saved_paths: list[str] = []
    for idx, (score, src_path) in enumerate(selected):
        dest = frames_dir / f"frame_{idx:03d}.jpg"
        shutil.copy2(src_path, dest)
        saved_paths.append(str(dest))
        log.debug(f"  frame {idx:03d}: score={score:.3f}")

    # ── GDPR cleanup: delete downloaded videos + temp frames ──────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.info(f"[{user_id}] Temp YouTube videos deleted (GDPR)")

    self.update_state(state="PROGRESS", meta={"stage": "done", "progress_pct": 100})
    log.info(f"[{user_id}] Reference frames ready: {len(saved_paths)} frames in {frames_dir}")

    return {
        "status": "completed",
        "user_id": user_id,
        "frame_count": len(saved_paths),
        "frames_dir": str(frames_dir),
        "progress_pct": 100,
    }


@celery_app.task(
    bind=True,
    name="backend.workers.training_worker.run_lora_training",
    soft_time_limit=1200,   # 20 min: prep only (download + clip + zip + submit)
    time_limit=1260,
)
def run_lora_training(
    self,
    user_id: str,
    youtube_urls: list[str] = [],
):
    """
    Full LoRA training pipeline:
      1. Use uploaded videos (and optionally YouTube URLs)
      2. Segment into face clips
      3. Zip clips
      4. Create destination model in user's Replicate account
      5. Start Replicate training — uploaded videos deleted after zip is committed
      6. Poll until done, store version_id in DB
    """
    from backend.services import replicate_service, storage_service
    from backend.db.database import SessionLocal
    from backend.db import crud

    tmp_dir = Path("/tmp/stylevid2/tmp") / f"lora_{user_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    trigger_word = f"SBJT{user_id.replace('-','')[:6].upper()}"

    def _prog(pct: int, msg: str):
        log.info(f"[lora/{user_id}] {pct}%  {msg}")
        self.update_state(state="PROGRESS", meta={"progress_pct": pct, "stage": msg})

    try:
        # ── 1. Resolve Replicate username ─────────────────────────────────────
        _prog(3, "Resolving Replicate account…")
        db = SessionLocal()
        try:
            user = crud.get_user_by_id(db, user_id)
            if not user or not user.replicate_key_encrypted:
                raise RuntimeError("Replicate key is not configured for this account.")
            replicate_key = decrypt_key(user.replicate_key_encrypted)
            username = user.replicate_username if user else None
            if not username:
                username = replicate_service.get_replicate_username(replicate_key)
                crud.set_replicate_username(db, user_id, username)
        finally:
            db.close()

        # ── 2. Collect videos (uploaded files + optional YouTube URLs) ───────
        _prog(8, "Collecting training videos…")
        downloaded: list[Path] = []
        for url in youtube_urls:
            p = _download_video(url, tmp_dir)
            if p:
                downloaded.append(p)
        for local_vid in storage_service.training_video_paths(user_id):
            downloaded.append(local_vid)
            log.info(f"[lora/{user_id}] Using uploaded video: {local_vid.name}")
        if not downloaded:
            raise RuntimeError(
                "No videos available. Upload video files using the 'Boost Quality' section."
            )

        # Normalize AV1 videos to H.264 for reliable clip extraction.
        downloaded = [_normalize_video_for_processing(v, tmp_dir, user_id) for v in downloaded]

        log.info(f"[lora/{user_id}] {len(downloaded)} video(s) ready for processing")

        # ── 3. Segment into face clips ────────────────────────────────────────
        _prog(20, "Finding face segments…")
        clips_dir = tmp_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        all_clips: list[Path] = []
        for vid in downloaded:
            clips = _segment_face_clips(vid, clips_dir)
            all_clips.extend(clips)
            log.info(f"[lora/{user_id}] {vid.name} → {len(clips)} clips")

        if not all_clips:
            raise RuntimeError(
                "No usable face segments found. "
                "Re-run Extract Frames first to verify your videos contain clear face footage."
            )

        # Cap at max clips, prefer even spread across videos
        all_clips = all_clips[:_MAX_CLIPS]
        log.info(f"[lora/{user_id}] {len(all_clips)} face clips selected for training")

        # ── 4. Zip clips ──────────────────────────────────────────────────────
        _prog(40, f"Packaging {len(all_clips)} clips…")
        zip_path = storage_service.training_zip_path(user_id)
        _zip_clips(all_clips, zip_path)
        zip_mb = zip_path.stat().st_size / 1e6
        log.info(f"[lora/{user_id}] Training zip: {zip_mb:.1f} MB")

        # ── 5. Create destination model ───────────────────────────────────────
        _prog(45, "Creating destination model in your Replicate account…")
        model_name = f"stylevid-{user_id.replace('-','')[:8]}"
        try:
            model_ref = replicate_service.ensure_destination_model(
                replicate_key, username, model_name
            )

            # ── 6. Start training ─────────────────────────────────────────────
            _prog(50, f"Starting LoRA training on Replicate (~10–15 min, ~$5–10)…")
            training_id = replicate_service.start_lora_training(
                replicate_key, model_ref, zip_path, trigger_word,
                steps=_LORA_TRAIN_STEPS,
            )
        except Exception as exc:
            log.error(
                f"[lora/{user_id}] Replicate setup/training submission failed: {exc}",
                exc_info=True,
            )
            raise RuntimeError(
                "We couldn't start LoRA training with your Replicate account. "
                "Please check your Replicate key and try again."
            ) from exc

        # Store training_id immediately so it's retrievable if worker crashes
        db = SessionLocal()
        try:
            crud.set_lora_training_started(db, user_id, training_id, model_ref, trigger_word)
        finally:
            db.close()

        # Delete zip and uploaded training videos — no longer needed
        zip_path.unlink(missing_ok=True)
        storage_service.delete_training_videos(user_id)

        # ── 7. Hand off to non-blocking poller ───────────────────────────────
        _prog(55, "Training submitted — polling starts in 60 s…")
        check_lora_training.apply_async(
            args=[user_id, training_id],
            countdown=60,
            queue="generation",
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "status":       "submitted",
        "model_ref":    model_ref,
        "trigger_word": trigger_word,
        "progress_pct": 55,
    }


@celery_app.task(
    bind=True,
    name="backend.workers.training_worker.check_lora_training",
    max_retries=120,        # 120 × 60 s = 2 hours max
    default_retry_delay=60,
)
def check_lora_training(self, user_id: str, training_id: str) -> dict:
    """
    Non-blocking Replicate training poller. Called every 60 s via self.retry()
    until the training succeeds, fails, or the 2-hour cap is reached.
    """
    from backend.services import replicate_service
    from backend.db.database import SessionLocal
    from backend.db import crud

    db = SessionLocal()
    try:
        user = crud.get_user_by_id(db, user_id)
        if not user or not user.replicate_key_encrypted:
            log.error(f"[check_lora/{user_id}] User or key not found — aborting poll")
            return {"status": "error"}
        if user.lora_status not in ("pending", "running"):
            log.info(f"[check_lora/{user_id}] Status already '{user.lora_status}' — skipping poll")
            return {"status": user.lora_status}
        replicate_key = decrypt_key(user.replicate_key_encrypted)
    finally:
        db.close()

    result = replicate_service.poll_lora_training(replicate_key, training_id)
    status = result["status"]
    attempt = self.request.retries
    # Rough progress estimate: 55% at submission, creeps toward 95% over 2 hours
    pct = min(55 + int(attempt / 120 * 40), 95)
    log.info(f"[check_lora/{user_id}] {pct}%  Replicate status: {status}  (poll #{attempt})")

    if status == "succeeded":
        version_id  = result.get("version_id")
        weights_url = result.get("weights_url")
        if not version_id:
            db = SessionLocal()
            try:
                crud.set_lora_training_failed(db, user_id, "Training succeeded but no version_id returned.")
            finally:
                db.close()
            return {"status": "failed"}
        db = SessionLocal()
        try:
            crud.set_lora_training_complete(db, user_id, version_id, weights_url)
        finally:
            db.close()
        log.info(f"[check_lora/{user_id}] Complete! version={version_id}  weights={'yes' if weights_url else 'no'}")
        return {"status": "completed", "version_id": version_id, "weights_url": weights_url}

    if status in ("failed", "canceled"):
        err = result.get("error") or f"Replicate training {status}."
        db = SessionLocal()
        try:
            crud.set_lora_training_failed(db, user_id, err)
        finally:
            db.close()
        log.error(f"[check_lora/{user_id}] Training {status}: {err}")
        return {"status": "failed", "error": err}

    # Still running — retry after 60 s
    raise self.retry(countdown=60)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _video_codec(video_path: Path) -> str:
    """Return primary video codec name (e.g. h264, av1), or empty string on failure."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return ""
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            return ""
        return str(streams[0].get("codec_name") or "").strip().lower()
    except Exception:
        return ""


def _normalize_video_for_processing(video_path: Path, tmp_dir: Path, user_id: str) -> Path:
    """
    Normalize AV1 videos to H.264 so downstream OpenCV/InsightFace processing is stable.
    Returns original path if no conversion is needed or conversion fails.
    """
    codec = _video_codec(video_path)
    if codec != "av1":
        return video_path

    norm_dir = tmp_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    out_path = norm_dir / f"{video_path.stem}_h264.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            log.info(f"[{user_id}] Normalized AV1 video to H.264: {video_path.name} -> {out_path.name}")
            return out_path
        log.warning(
            f"[{user_id}] AV1 normalization failed for {video_path.name}; using original file. "
            f"ffmpeg rc={result.returncode}"
        )
        return video_path
    except Exception as exc:
        log.warning(f"[{user_id}] AV1 normalization error for {video_path.name}: {exc}")
        return video_path

def _download_video(url: str, dest_dir: Path) -> Path | None:
    """
    Download one YouTube video (including Shorts). Returns path or None on failure.

    Format note: omit [ext=mp4] filter — Shorts often only have webm streams.
    --merge-output-format mp4 re-muxes everything to mp4 regardless.
    """
    import tempfile
    
    out_template = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--output", out_template,
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--no-warnings",
    ]
    
    # In production YouTube blocks server IPs without browser cookies.
    # Support two approaches:
    # 1. YT_DLP_COOKIES_FILE: path to Netscape-format cookies.txt
    # 2. YT_DLP_COOKIES: raw cookie string (format: "name1=value1; name2=value2; ...")
    temp_cookie_file = None
    try:
        # Approach 1: Use mounted cookies file if available
        cookies_file = settings.yt_dlp_cookies_file
        if cookies_file and Path(cookies_file).exists():
            cmd += ["--cookies", cookies_file]
            log.debug(f"Using cookies file: {cookies_file}")
        # Approach 2: Create temp cookies file from raw cookie string
        elif settings.yt_dlp_cookies:
            temp_cookie_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
            log.debug(f"Creating temp cookie file from YT_DLP_COOKIES env var")
            
            # Convert raw cookie string to Netscape format
            # Input format: "name1=value1; name2=value2"
            # Netscape format: domain flag path secure expiration name value
            netscape_header = "# Netscape HTTP Cookie File\n"
            temp_cookie_file.write(netscape_header)
            
            cookie_count = 0
            for pair in settings.yt_dlp_cookies.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    name, value = pair.split("=", 1)
                    name = name.strip()
                    value = value.strip()
                    # Netscape format: domain flag path secure expiration name value
                    # Tab-separated fields
                    netscape_line = f"youtube.com\tTRUE\t/\tTRUE\t0\t{name}\t{value}\n"
                    temp_cookie_file.write(netscape_line)
                    cookie_count += 1
                    log.debug(f"Added cookie: {name}")
            
            temp_cookie_file.close()
            log.debug(f"Wrote {cookie_count} cookies to {temp_cookie_file.name}")
            cmd += ["--cookies", temp_cookie_file.name]
        else:
            log.warning(f"No cookies configured (YT_DLP_COOKIES_FILE or YT_DLP_COOKIES env vars not set)")
        
        cmd.append(url)
        result = subprocess.run(cmd, timeout=600, capture_output=True, text=True)
        if result.returncode not in (0, 1):  # 1 = partial/warning, still usable
            log.warning(f"yt-dlp exited {result.returncode} for {url}:\n{result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        log.warning(f"yt-dlp timed out for {url}")
        return None
    except Exception as exc:
        log.warning(f"yt-dlp error for {url}: {exc}")
        return None
    finally:
        # Clean up temp cookie file if created
        if temp_cookie_file:
            try:
                Path(temp_cookie_file.name).unlink(missing_ok=True)
            except Exception as e:
                log.debug(f"Failed to clean up temp cookie file: {e}")

    mp4s = list(dest_dir.glob("*.mp4"))
    if not mp4s:
        log.warning(f"yt-dlp produced no mp4 for {url}. stderr: {result.stderr[-300:]}")
        return None
    return mp4s[-1]


def _sample_frames(video_path: Path, out_dir: Path, sample_every_n_sec: int = 2) -> list[Path]:
    """Extract one frame every N seconds from the video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    interval = max(1, int(fps * sample_every_n_sec))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[Path] = []
    idx = 0
    frame_num = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_num % interval == 0:
            p = out_dir / f"{video_path.stem}_f{frame_num:06d}.jpg"
            cv2.imwrite(str(p), frame)
            frames.append(p)
            idx += 1
        frame_num += 1

    cap.release()
    log.debug(f"Sampled {len(frames)} frames from {video_path.name} ({total} total frames)")
    return frames


def _score_faces(frame_paths: list[Path]) -> list[tuple[float, Path]]:
    """
    Score each frame by face quality. Returns (score, path) pairs for frames
    that pass the minimum face score threshold.
    """
    from insightface.app import FaceAnalysis
    import numpy as np
    import cv2

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    results: list[tuple[float, Path]] = []
    for path in frame_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            path.unlink(missing_ok=True)
            continue

        best = max(faces, key=lambda f: f.det_score)
        if best.det_score < settings.min_face_score:
            path.unlink(missing_ok=True)
            continue

        # Score = detection confidence × relative face size
        h, w = img.shape[:2]
        bbox = best.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        frame_area = w * h
        size_score = min(face_area / frame_area * 10, 1.0)  # normalize; cap at 1

        # Penalize extreme head pose (yaw/pitch beyond ±30°)
        pose = best.pose if hasattr(best, "pose") else np.zeros(3)
        yaw, pitch = abs(pose[1]), abs(pose[0])
        pose_penalty = max(0, 1.0 - (max(yaw, pitch) / 45.0))

        score = float(best.det_score) * 0.5 + size_score * 0.3 + pose_penalty * 0.2
        results.append((score, path))

    return results


def _segment_face_clips(video_path: Path, out_dir: Path) -> list[Path]:
    """
    Scan a video and extract short clips where a high-quality face is visible.

    Uses InsightFace for face detection + quality scoring.
    Clips are extracted with ffmpeg (no audio, libx264).
    Returns list of extracted clip paths.
    """
    import cv2
    from insightface.app import FaceAnalysis
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps

    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Scan every _SCAN_INTERVAL seconds for a good face frame
    good_timestamps: list[float] = []
    t = 0.0
    while t < total_duration:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            t += _SCAN_INTERVAL
            continue

        faces = face_app.get(frame)
        if faces:
            best = max(faces, key=lambda f: f.det_score)
            if best.det_score >= _MIN_FACE_SCORE:
                bbox = best.bbox
                face_w = bbox[2] - bbox[0]
                face_h = bbox[3] - bbox[1]
                if min(face_w, face_h) >= _MIN_FACE_PX:
                    good_timestamps.append(t)
        t += _SCAN_INTERVAL

    cap.release()

    if not good_timestamps:
        log.warning(f"No qualifying face moments in {video_path.name}")
        return []

    # Group timestamps into non-overlapping clips
    clip_specs: list[tuple[float, float]] = []
    last_clip_end = -_CLIP_DURATION
    for ts in good_timestamps:
        if ts < last_clip_end:
            continue
        start = max(0.0, ts - 0.5)
        end   = min(total_duration, start + _CLIP_DURATION)
        if end - start < 2.0:
            continue
        clip_specs.append((start, end))
        last_clip_end = end
        if len(clip_specs) >= _MAX_CLIPS:
            break

    # Extract clips with ffmpeg
    extracted: list[Path] = []
    for i, (start, end) in enumerate(clip_specs):
        out_path = out_dir / f"{video_path.stem}_clip{i:03d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-t",  f"{end - start:.3f}",
            "-i",  str(video_path),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-an",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            extracted.append(out_path)
        else:
            log.warning(f"ffmpeg clip extraction failed for segment {i} of {video_path.name}")

    log.debug(f"{video_path.name}: {len(clip_specs)} segments → {len(extracted)} clips extracted")
    return extracted


def _zip_clips(clip_paths: list[Path], zip_path: Path) -> None:
    """Zip all clip files into a single archive for Replicate upload."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in clip_paths:
            zf.write(p, arcname=p.name)
    log.info(f"Zipped {len(clip_paths)} clips → {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
