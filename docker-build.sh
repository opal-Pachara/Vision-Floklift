#!/usr/bin/env bash
set -e

IMAGE_NAME="yolo-safety-monitor:latest"

echo "=================================================================="
echo "  Building Docker Image with Embedded YOLOv12n Model"
echo "  Target: ${IMAGE_NAME}"
echo "=================================================================="

docker build -t "${IMAGE_NAME}" .

echo "=================================================================="
echo "  Docker Build Finished Successfully!"
echo "  Run via: ./docker-run.sh or 'docker compose up -d'"
echo "=================================================================="
