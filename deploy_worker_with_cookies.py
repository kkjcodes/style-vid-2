#!/usr/bin/env python3
"""Deploy worker container with YT_DLP_COOKIES support."""
import os
import subprocess
import json
import time

RG = "stylevid-rg"
WORKER_APP = "stylevid-worker"
CONTAINER_ENV = "managedEnvironment-stylevidrg-b46c"
ACR_NAME = "stylevidacr"
REDIS_NAME = "stylevidredis"


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# Get registry password
print("Getting registry credentials...")
REGISTRY_PWD = subprocess.run(
    ['az', 'acr', 'credential', 'show', '--resource-group', RG, '--name', ACR_NAME,
     '--query', 'passwords[0].value', '-o', 'tsv'],
    capture_output=True, text=True, check=True, timeout=15
).stdout.strip()

# Get Redis key
print("Getting Redis key...")
REDIS_KEY = subprocess.run(
    ['az', 'redis', 'list-keys', '-g', RG, '-n', REDIS_NAME, '-o', 'json'],
    capture_output=True, text=True, check=True, timeout=15
)
REDIS_KEY = json.loads(REDIS_KEY.stdout)['primaryKey']
REDIS_HOST = subprocess.run(
    ['az', 'redis', 'show', '-g', RG, '-n', REDIS_NAME, '--query', 'hostName', '-o', 'tsv'],
    capture_output=True, text=True, check=True, timeout=15
).stdout.strip()

DATABASE_URL = _required_env("DATABASE_URL")
ENCRYPTION_KEY = _required_env("ENCRYPTION_KEY")
JWT_SECRET = _required_env("JWT_SECRET")
YT_DLP_COOKIES = os.getenv("YT_DLP_COOKIES", "")

print("Restarting Worker with YT_DLP_COOKIES support...")

# Delete existing worker
print("Deleting existing worker...")
subprocess.run(['az', 'containerapp', 'delete', '-n', WORKER_APP, '-g', RG, '--yes'],
               capture_output=True, timeout=60)
time.sleep(3)

# Create worker with YT_DLP_COOKIES env var
print("Creating worker container...")
worker_result = subprocess.run([
    'az', 'containerapp', 'create',
    '--name', WORKER_APP,
    '--resource-group', RG,
    '--environment', CONTAINER_ENV,
    '--image', f'{ACR_NAME}.azurecr.io/stylevid:latest',
    '--registry-server', f'{ACR_NAME}.azurecr.io',
    '--registry-username', ACR_NAME,
    '--registry-password', REGISTRY_PWD,
    '--env-vars',
    'APP_ENV=production',
    f'DATABASE_URL={DATABASE_URL}',
    f'ENCRYPTION_KEY={ENCRYPTION_KEY}',
    f'JWT_SECRET={JWT_SECRET}',
    'LOCAL_STORAGE_DIR=/tmp/stylevid2',
    f'REDIS_URL=rediss://:{REDIS_KEY}@{REDIS_HOST}:6380/0?ssl_cert_reqs=CERT_REQUIRED',
    'SERVICE_MODE=worker',
    f'YT_DLP_COOKIES={YT_DLP_COOKIES}',
    '--cpu', '0.5',
    '--memory', '1.0',
    '--min-replicas', '1',
    '--max-replicas', '5'
], capture_output=True, text=True, timeout=120)

if worker_result.returncode == 0:
    print("✓ Worker container created with YT_DLP_COOKIES support!")
else:
    print(f"❌ Error: {worker_result.stderr[:300]}")
    exit(1)

time.sleep(10)

# Check status
print("\nChecking worker status...")
status_result = subprocess.run(
    ['az', 'containerapp', 'show', '-n', 'stylevid-worker', '-g', RG, '-o', 'json'],
    capture_output=True, text=True, timeout=15
)
if status_result.returncode == 0:
    app = json.loads(status_result.stdout)
    print(f"Worker Status: {app['properties'].get('runningStatus')}")
    print(f"Provisioning: {app['properties'].get('provisioningState')}")
else:
    print("❌ Query failed")
    exit(1)

print("\n✓ Deployment complete!")
