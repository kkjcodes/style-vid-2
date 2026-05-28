from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    secret_key: str = "local-dev-secret"

    # Model tracks
    wan_model_track: str = "i2v"          # "i2v" | "animate"
    wan_i2v_model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    wan_animate_model_id: str = "Wan-AI/Wan2.2-Animate-14B-Diffusers"

    # Storage
    local_storage_dir: str = "/tmp/stylevid2"

    # Queue
    redis_url: str = "redis://localhost:6379/0"
    # Production DB (preferred): postgresql+psycopg2://user:pass@host:5432/db?sslmode=require
    # If unset, app falls back to sqlite_db_url for local/dev.
    database_url: str = ""

    # Generation limits
    max_num_frames: int = 81        # 4*20+1 — ~5s at 16fps
    max_video_seconds: int = 15     # hard cap per request

    # Speed knobs
    wan_num_inference_steps: int = 20   # 50 default → ~2.5h; 20 → ~58min on MPS
    wan_int8: bool = True               # int8 quantize transformer; halves VRAM, enables full-MPS run
    wan_max_area: int = 76800           # 320×240 portrait; 163840 (320×512) OOMs on 48 GB MPS during VAE decode

    # ── Replicate models ─────────────────────────────────────────────────────
    # Selfie mode: wan-2.7-i2v — first_frame + first_clip continuation, up to 15s
    replicate_i2v_model: str = "wan-video/wan-2.7-i2v"
    # Reference mode: wan-2.7-r2v — reference_images[] from YouTube frames, no training
    replicate_r2v_model: str = "wan-video/wan-2.7-r2v"
    # Generation defaults
    replicate_clip_duration: int = 10                          # seconds per clip (2-15 for i2v, 2-10 for r2v)
    replicate_resolution: str = "720p"                         # "720p" | "1080p"
    replicate_aspect_ratio: str = "9:16"                       # portrait for selfie/creator videos
    replicate_clips_per_video: int = 2                         # 2 × 10s = 20s default
    # Reference frame extraction
    max_reference_frames: int = 20                             # frames extracted from YouTube
    min_face_score: float = 0.65                               # InsightFace det_score threshold

    # ── Security ─────────────────────────────────────────────────────────────
    # Fernet key for encrypting Replicate API keys at rest.
    # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # Set via ENCRYPTION_KEY env var in production.
    encryption_key: str = "REPLACE_ME_IN_PROD_USE_ENV_VAR_32B"
    jwt_secret: str = "REPLACE_ME_IN_PROD_jwt_secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 720                                # 30 days

    # ── Storage ──────────────────────────────────────────────────────────────
    blob_storage_dir: str = "/tmp/stylevid2/blob"
    sqlite_db_url: str = "sqlite:////tmp/stylevid2/app.db"

    # ── Azure Blob Storage (prod video serving) ───────────────────────────────
    # Set AZURE_STORAGE_CONNECTION_STRING env var in production.
    # If empty, videos are served directly from container disk (dev mode).
    azure_storage_connection_string: str = ""
    azure_storage_container: str = "stylevid-videos"

    # ── yt-dlp cookies (required in prod to bypass YouTube bot detection) ────────
    # Option 1: Mount a cookies.txt file (Netscape format) and set the path
    yt_dlp_cookies_file: str = ""
    # Option 2: Pass cookies as raw string. Format: "name1=value1; name2=value2; ..." (newline-separated pairs)
    # Key cookies needed for YouTube Shorts:
    #   __Secure-3PSIDTS — Main tracking/session cookie (CRITICAL)
    #   PREF — User preferences
    #   CONSENT — GDPR consent status (set to "YES")
    # To extract: Open https://youtube.com in logged-in browser → DevTools Network tab → 
    #   Copy request header "Cookie:" or use browser extension "Get cookies.txt LOCALLY"
    yt_dlp_cookies: str = ""

    # ── App URL (used in password-reset emails) ───────────────────────────────
    app_url: str = "http://localhost:8000"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated string into list if needed."""
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    # ── SMTP (password reset emails) ─────────────────────────────────────────
    # Leave SMTP_HOST empty to disable email sending (reset links logged instead)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
