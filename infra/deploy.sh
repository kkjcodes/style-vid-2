#!/usr/bin/env bash
# One-command Azure deployment for StyleVid.
# Run once to provision all resources; subsequent deploys go through GitHub Actions.
#
# Prerequisites:
#   az login
#   Set variables below or export them as env vars.

set -euo pipefail

# ── Config — set these ─────────────────────────────────────────────────────────
RG="${AZURE_RG:-stylevid-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-stylevidacr}"
APP_ENV_NAME="${APP_ENV_NAME:-stylevid-env}"
IMAGE="${ACR_NAME}.azurecr.io/stylevid:latest"
REDIS_NAME="${REDIS_NAME:-stylevid-redis}"
API_APP_NAME="${API_APP_NAME:-stylevid-api}"
WORKER_APP_NAME="${WORKER_APP_NAME:-stylevid-worker}"

PG_SERVER_NAME="${PG_SERVER_NAME:-stylevid-pg}"
PG_DB_NAME="${PG_DB_NAME:-stylevid}"
PG_ADMIN_USER="${PG_ADMIN_USER:-stylevidadmin}"

STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-stylevidstorage}"
FILE_SHARE_NAME="${FILE_SHARE_NAME:-stylevid-files}"
STORAGE_NAME_IN_ENV="${STORAGE_NAME_IN_ENV:-stylevidfiles}"
VOLUME_NAME="${VOLUME_NAME:-stylevidfilesvol}"

# ── Secrets — set these as env vars, never hardcode ───────────────────────────
ENCRYPTION_KEY="${ENCRYPTION_KEY:?Set ENCRYPTION_KEY}"
JWT_SECRET="${JWT_SECRET:?Set JWT_SECRET}"
PG_ADMIN_PASSWORD="${PG_ADMIN_PASSWORD:?Set PG_ADMIN_PASSWORD}"

# Optional (password reset email)
SMTP_HOST="${SMTP_HOST:-}"
SMTP_PORT="${SMTP_PORT:-587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASSWORD="${SMTP_PASSWORD:-}"
SMTP_FROM="${SMTP_FROM:-}"

echo "==> Creating resource group: $RG in $LOCATION"
az group create --name "$RG" --location "$LOCATION"

echo "==> Creating Azure Container Registry: $ACR_NAME"
az acr create --resource-group "$RG" --name "$ACR_NAME" --sku Basic --admin-enabled true

ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo "==> Creating Azure Cache for Redis (Basic C0)"
az redis create \
  --resource-group "$RG" \
  --name "$REDIS_NAME" \
  --location "$LOCATION" \
  --sku Basic \
  --vm-size c0

REDIS_KEY=$(az redis list-keys --resource-group "$RG" --name "$REDIS_NAME" --query primaryKey -o tsv)
REDIS_HOST=$(az redis show --resource-group "$RG" --name "$REDIS_NAME" --query hostName -o tsv)
REDIS_URL="rediss://:${REDIS_KEY}@${REDIS_HOST}:6380/0"

echo "==> Creating Azure PostgreSQL Flexible Server"
az postgres flexible-server create \
  --resource-group "$RG" \
  --name "$PG_SERVER_NAME" \
  --location "$LOCATION" \
  --admin-user "$PG_ADMIN_USER" \
  --admin-password "$PG_ADMIN_PASSWORD" \
  --sku-name Standard_B1ms \
  --tier Burstable \
  --version 16 \
  --storage-size 32 \
  --yes

echo "==> Creating PostgreSQL database: $PG_DB_NAME"
az postgres flexible-server db create \
  --resource-group "$RG" \
  --server-name "$PG_SERVER_NAME" \
  --database-name "$PG_DB_NAME"

# Allow Azure services to connect to Postgres.
az postgres flexible-server firewall-rule create \
  --resource-group "$RG" \
  --name "$PG_SERVER_NAME" \
  --rule-name allow-azure \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0

PG_HOST=$(az postgres flexible-server show --resource-group "$RG" --name "$PG_SERVER_NAME" --query fullyQualifiedDomainName -o tsv)
DATABASE_URL="postgresql+psycopg2://${PG_ADMIN_USER}:${PG_ADMIN_PASSWORD}@${PG_HOST}:5432/${PG_DB_NAME}?sslmode=require"

echo "==> Creating Storage Account + Azure File Share"
az storage account create \
  --resource-group "$RG" \
  --name "$STORAGE_ACCOUNT" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2

STORAGE_KEY=$(az storage account keys list --resource-group "$RG" --account-name "$STORAGE_ACCOUNT" --query "[0].value" -o tsv)

az storage share create \
  --name "$FILE_SHARE_NAME" \
  --account-name "$STORAGE_ACCOUNT" \
  --account-key "$STORAGE_KEY"

echo "==> Creating Container Apps environment"
az containerapp env create \
  --name "$APP_ENV_NAME" \
  --resource-group "$RG" \
  --location "$LOCATION"

echo "==> Registering Azure Files storage with Container Apps environment"
az containerapp env storage set \
  --name "$APP_ENV_NAME" \
  --resource-group "$RG" \
  --storage-name "$STORAGE_NAME_IN_ENV" \
  --azure-file-account-name "$STORAGE_ACCOUNT" \
  --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "$FILE_SHARE_NAME" \
  --access-mode ReadWrite

# Helper list for optional SMTP vars
SMTP_ENV_ARGS=()
if [[ -n "$SMTP_HOST" ]]; then
  SMTP_ENV_ARGS+=("SMTP_HOST=${SMTP_HOST}")
  SMTP_ENV_ARGS+=("SMTP_PORT=${SMTP_PORT}")
  SMTP_ENV_ARGS+=("SMTP_USER=${SMTP_USER}")
  SMTP_ENV_ARGS+=("SMTP_PASSWORD=${SMTP_PASSWORD}")
  SMTP_ENV_ARGS+=("SMTP_FROM=${SMTP_FROM}")
fi

echo "==> Deploying API container app"
az containerapp create \
  --name "$API_APP_NAME" \
  --resource-group "$RG" \
  --environment "$APP_ENV_NAME" \
  --image "$IMAGE" \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --registry-username "$ACR_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --ingress external \
  --target-port 8000 \
  --min-replicas 0 \
  --max-replicas 5 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --volumes "${VOLUME_NAME}=azurefile,${STORAGE_NAME_IN_ENV}" \
  --volume-mounts "/tmp/stylevid2=${VOLUME_NAME}" \
  --env-vars \
    "APP_ENV=production" \
    "REDIS_URL=${REDIS_URL}" \
    "DATABASE_URL=${DATABASE_URL}" \
    "ENCRYPTION_KEY=${ENCRYPTION_KEY}" \
    "JWT_SECRET=${JWT_SECRET}" \
    "LOCAL_STORAGE_DIR=/tmp/stylevid2" \
    "CORS_ORIGINS=" \
    "APP_URL=" \
    "${SMTP_ENV_ARGS[@]}"

echo "==> Deploying Worker container app"
az containerapp create \
  --name "$WORKER_APP_NAME" \
  --resource-group "$RG" \
  --environment "$APP_ENV_NAME" \
  --image "$IMAGE" \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --registry-username "$ACR_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --command "celery" \
  --args "-A,backend.workers.celery_app:celery_app,worker,--loglevel=info,-Q,generation,-c,2" \
  --ingress disabled \
  --min-replicas 0 \
  --max-replicas 10 \
  --cpu 2.0 \
  --memory 4.0Gi \
  --volumes "${VOLUME_NAME}=azurefile,${STORAGE_NAME_IN_ENV}" \
  --volume-mounts "/tmp/stylevid2=${VOLUME_NAME}" \
  --env-vars \
    "APP_ENV=production" \
    "REDIS_URL=${REDIS_URL}" \
    "DATABASE_URL=${DATABASE_URL}" \
    "ENCRYPTION_KEY=${ENCRYPTION_KEY}" \
    "JWT_SECRET=${JWT_SECRET}" \
    "LOCAL_STORAGE_DIR=/tmp/stylevid2" \
    "${SMTP_ENV_ARGS[@]}"

API_FQDN=$(az containerapp show --name "$API_APP_NAME" --resource-group "$RG" --query "properties.configuration.ingress.fqdn" -o tsv)
API_URL="https://${API_FQDN}"

echo "==> Updating runtime URL and CORS settings"
az containerapp update \
  --name "$API_APP_NAME" \
  --resource-group "$RG" \
  --set-env-vars "APP_URL=${API_URL}" "CORS_ORIGINS=${API_URL}"

echo ""
echo "✓ Deployment complete!"
echo "  API: ${API_URL}"
echo ""
echo "Next steps:"
echo "  1. Build and push container image to ${ACR_NAME}.azurecr.io/stylevid:latest"
echo "  2. Verify API health: ${API_URL}/health"
echo "  3. Add GitHub secrets for CI/CD rollout"
