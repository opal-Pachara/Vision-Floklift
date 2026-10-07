#!/usr/bin/env bash
set -e

echo "======================================================"
echo "  Building YOLO C++ Inference Engine"
echo "======================================================"

# Determine number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    NUM_JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
    NUM_JOBS=$(nproc 2>/dev/null || echo 4)
fi

mkdir -p build
cd build

echo "Configuring CMake (Release Mode)..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Compiling project using ${NUM_JOBS} cores..."
cmake --build . -j"${NUM_JOBS}"

echo "======================================================"
echo "  Build successful! Executable is at: ./build/yolo_cpp"
echo "======================================================"
echo "Quick Test:"
echo "  ./build/yolo_cpp --benchmark 50"
echo "  ./build/yolo_cpp --image <path_to_image.jpg>"
echo "======================================================"
