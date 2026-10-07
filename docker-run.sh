#!/usr/bin/env bash
set -e

PORT=${1:-8080}
CONTAINER_NAME="yolo-safety-monitoring"
IMAGE_NAME="yolo-safety-monitor:latest"

echo "=================================================================="
echo "  Deploying YOLOv12n AI Safety Monitoring System via Docker"
echo "  Port: ${PORT}"
echo "=================================================================="

# Stop and remove existing container if running
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "[Docker] Stopping previous container ${CONTAINER_NAME}..."
    docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
fi

docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8080 \
    --shm-size=2g \
    --restart unless-stopped \
    -v "$(pwd)/config:/app/config" \
    -v "$(pwd)/results:/app/results" \
    ${IMAGE_NAME}

echo "=================================================================="
echo "  Container started successfully!"
echo "  Access Web App at: http://localhost:${PORT}"
echo "  View logs: docker logs -f ${CONTAINER_NAME}"
echo "=================================================================="
