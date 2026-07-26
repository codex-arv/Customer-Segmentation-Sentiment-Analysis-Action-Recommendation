#!/bin/sh

set -e

echo ""
echo "==========================================================================="
echo " STARTING CUSTOMER SEGMENTATION API CONTAINER"
echo "==========================================================================="

echo ""
echo "Checking environment..."

echo "PORT = ${PORT:-10000}"
echo "ARTIFACT DIRECTORY = ${ARTIFACT_DIR:-/app/artifacts}"

echo ""
echo "Starting FastAPI immediately..."
echo "Artifact download and model loading will run in the background."

exec uvicorn main:app \
    --host 0.0.0.0 \
    --port "${PORT:-10000}"