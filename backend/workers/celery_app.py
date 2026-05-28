from celery import Celery
from backend.core.config import get_settings
import ssl

settings = get_settings()

celery_app = Celery(
    "stylevid2",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "backend.workers.generation_worker",
        "backend.workers.pipeline_worker",
        "backend.workers.training_worker",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_routes={
        "backend.workers.generation_worker.*": {"queue": "generation"},
        "backend.workers.pipeline_worker.*":   {"queue": "generation"},
        "backend.workers.training_worker.*":   {"queue": "generation"},
    },
)
