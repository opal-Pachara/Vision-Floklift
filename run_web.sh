#!/usr/bin/env bash
set -e

PORT=${1:-8080}
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "=================================================================="
echo "  YOLOv12n Industrial AI Safety Monitoring Web Application"
echo "  Engine: C++ / ONNX Runtime / OpenCV"
echo "  Target Port: ${PORT}"
echo "=================================================================="

# 1. Ensure C++ shared library exists
if [[ ! -f "build/libyolo_detector.dylib" && ! -f "build/libyolo_detector.so" ]]; then
    echo "[Engine] Shared library not found. Compiling C++ engine..."
    ./build.sh
fi

# 2. Check if ONNX model exists
if [[ ! -f "best.onnx" ]]; then
    echo "[Model] best.onnx not found. Exporting best.pt to ONNX..."
    python3 export_model.py
fi

# 3. Start Web Server
echo "[Server] Launching FastAPI Industrial Control Room on port ${PORT}..."
echo "[Server] Access Dashboard at: http://localhost:${PORT}"
python3 server.py "${PORT}"
