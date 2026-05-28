"""
Azure Blob Storage service for generated videos.

Enabled only when AZURE_STORAGE_CONNECTION_STRING is set.
In dev the service is a no-op and videos are served from local disk.
"""
from __future__ import annotations

import logging
from pathlib import Path

from backend.core.config import get_settings

log = logging.getLogger("blob_service")


def is_enabled() -> bool:
    return bool(get_settings().azure_storage_connection_string)


def upload_video(user_id: str, filename: str, local_path: Path) -> str:
    """Upload video to blob storage. Returns the public blob URL."""
    from azure.storage.blob import BlobServiceClient

    s = get_settings()
    client = BlobServiceClient.from_connection_string(s.azure_storage_connection_string)
    container = client.get_container_client(s.azure_storage_container)
    try:
        container.create_container(public_access="blob")
    except Exception:
        pass  # container already exists

    blob_name = f"users/{user_id}/outputs/{filename}"
    with open(local_path, "rb") as f:
        container.upload_blob(blob_name, f, overwrite=True)

    url = f"https://{client.account_name}.blob.core.windows.net/{s.azure_storage_container}/{blob_name}"
    log.info(f"Uploaded blob: {url}")
    return url


def delete_user_videos(user_id: str) -> int:
    """Delete all blobs for a user (GDPR). Returns count deleted."""
    from azure.storage.blob import BlobServiceClient

    s = get_settings()
    client = BlobServiceClient.from_connection_string(s.azure_storage_connection_string)
    container = client.get_container_client(s.azure_storage_container)
    prefix = f"users/{user_id}/"
    blobs = list(container.list_blobs(name_starts_with=prefix))
    for blob in blobs:
        container.delete_blob(blob.name)
    log.info(f"GDPR: deleted {len(blobs)} blobs for user={user_id}")
    return len(blobs)
