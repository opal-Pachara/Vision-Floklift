#!/usr/bin/env bash
set -e

echo "======================================================"
echo "    YOLO C++ Performance & Latency Benchmark"
echo "======================================================"

if [ ! -f "build/yolo_cpp" ]; then
    echo "Executable not found. Running build first..."
    ./build.sh
fi

RUNS=${1:-50}

echo ""
echo ">>> [1/2] Benchmarking Apple CoreML Acceleration (Apple Neural Engine / GPU) <<<"
./build/yolo_cpp --coreml --benchmark "${RUNS}"

echo ""
echo ">>> [2/2] Benchmarking CPU Multi-threaded Execution <<<"
./build/yolo_cpp --cpu --threads 4 --benchmark "${RUNS}"

echo "======================================================"
echo "Benchmark completed!"
echo "======================================================"
