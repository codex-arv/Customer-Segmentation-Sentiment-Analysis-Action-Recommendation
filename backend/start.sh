#!/bin/sh

set -e

echo ""
echo "==========================================================================="
echo " STARTING CUSTOMER SEGMENTATION API CONTAINER"
echo "==========================================================================="

echo ""
echo "[1/2] Checking environment..."
echo "PORT = ${PORT:-10000}"
echo "ARTIFACT DIRECTORY = ${ARTIFACT_DIR:-/app/artifacts}"

echo ""
echo "[2/2] Downloading artifacts from Cloudflare R2..."

python download_artifacts.py

echo ""
echo "Starting FastAPI..."

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT:-10000}"