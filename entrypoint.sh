#!/bin/bash
set -e

# Default to API if no mode specified
SERVICE_MODE="${SERVICE_MODE:-api}"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-1}"

case "$SERVICE_MODE" in
    api)
        exec uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
        ;;
    worker)
        exec celery -A backend.workers.celery_app:celery_app worker --loglevel=info -Q generation -c "$WORKER_CONCURRENCY" --prefetch-multiplier=1 --max-tasks-per-child=1
        ;;
    *)
        echo "Unknown SERVICE_MODE: $SERVICE_MODE"
        exit 1
        ;;
esac
