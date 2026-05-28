from __future__ import annotations
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.core.config import get_settings

settings = get_settings()


def _resolve_database_url() -> str:
    return settings.database_url or settings.sqlite_db_url


DB_URL = _resolve_database_url()
is_sqlite = DB_URL.startswith("sqlite")

# Ensure the directory exists before SQLite creates the file
if is_sqlite:
    Path(DB_URL.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False} if is_sqlite else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from backend.db import models  # noqa: F401 — registers models with Base
    from sqlalchemy import text

    Base.metadata.create_all(bind=engine)

    # PostgreSQL is managed via model metadata/alembic path. Skip SQLite ALTER fallbacks.
    if not is_sqlite:
        return

    # Safe additive migrations — add new columns without dropping existing data
    new_cols = [
        "replicate_username TEXT",
        "frames_training_id TEXT",
        "lora_model_ref TEXT",
        "lora_version_id TEXT",
        "lora_trigger_word TEXT",
        "lora_celery_job_id TEXT",
        "lora_training_id TEXT",
        "lora_weights_url TEXT",
        "lora_status TEXT",
        "lora_error TEXT",
        "email TEXT",
        "reset_token TEXT",
        "reset_token_expires TIMESTAMP",
    ]
    with engine.connect() as conn:
        for col_def in new_cols:
            col_name = col_def.split()[0]
            try:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_def}"))
                conn.commit()
            except Exception:
                pass  # column already exists
