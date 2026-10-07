#!/usr/bin/env bash
# Simple runner for live camera detection
CAM_ID=${1:-0}

if [ ! -f "build/run_camera" ]; then
    echo "Executable build/run_camera not found. Building now..."
    ./build.sh
fi

echo "Launching YOLO Camera Detector (Camera ID: $CAM_ID)..."
./build/run_camera "$CAM_ID"
