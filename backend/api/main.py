from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware

from backend.api.routes import onboard, generate, pipeline, training, me
from backend.api.routes import auth
from backend.api.limiter import limiter
from backend.core.logging_config import setup_logging

setup_logging()


_WEAK_SECRETS = {
    "REPLACE_ME_IN_PROD_USE_ENV_VAR_32B",
    "REPLACE_ME_IN_PROD_jwt_secret",
    "local-dev-secret",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.db.database import init_db
    from backend.core.config import get_settings
    import logging
    s = get_settings()
    log = logging.getLogger("startup")
    if s.app_env == "production":
        if s.encryption_key in _WEAK_SECRETS:
            raise RuntimeError("ENCRYPTION_KEY must be set to a strong random value in production.")
        if s.jwt_secret in _WEAK_SECRETS:
            raise RuntimeError("JWT_SECRET must be set to a strong random value in production.")
    elif s.encryption_key in _WEAK_SECRETS or s.jwt_secret in _WEAK_SECRETS:
        log.warning("⚠️  Using default secrets — set ENCRYPTION_KEY and JWT_SECRET before deploying.")
    init_db()
    yield


app = FastAPI(title="StyleVid", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

from backend.core.config import get_settings

s = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=s.cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Auth must be first
app.include_router(auth.router)
app.include_router(pipeline.router)
app.include_router(training.router)
app.include_router(me.router)
# Legacy routes kept for backward compat
app.include_router(onboard.router)
app.include_router(generate.router)

# Serve frontend
_frontend = Path(__file__).parent.parent.parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(_frontend / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
