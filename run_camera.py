#!/usr/bin/env python3
"""
Live Camera Detection Interface (Python Camera UI + High-Speed C++ Engine)
Runs YOLO model inference via compiled C++ shared library with CoreML/CPU acceleration,
while using Python OpenCV for camera streaming, keyboard events, and GUI display.
"""

import sys
import os
import time
import cv2
from yolo_cpp import YoloCppDetector

def main():
    camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("=" * 60)
    print(" YOLO Logistics Camera Detection")
    print(" Architecture: Python Camera UI + High-Performance C++ Engine")
    print("=" * 60)

    # 1. Initialize C++ Inference Engine
    print("\n[1/2] Initializing C++ Engine (best.onnx + Apple CoreML)...")
    try:
        detector = YoloCppDetector(
            model_path="best.onnx",
            labels_path="labels.txt",
            conf_thresh=0.25,
            nms_thresh=0.45,
            num_threads=4,
            use_coreml=True
        )
    except Exception as e:
        print(f"[Error] Failed to initialize C++ detector: {e}")
        print("Tip: Run './build.sh' to compile the C++ shared library first.")
        sys.exit(1)

    # 2. Open Camera in Python
    print(f"\n[2/2] Opening Camera (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened() and camera_id == 0:
        print("Camera 0 not available. Trying camera ID 1...")
        camera_id = 1
        cap.open(camera_id)

    if not cap.isOpened():
        print(f"\n[Error] Could not open camera {camera_id}.")
        print("Tip: Ensure camera permissions are granted in macOS System Settings > Privacy & Security > Camera.")
        return

    # Set camera resolution (720p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "YOLO Logistics Safety Detection (Python UI + C++ Engine)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "=" * 60)
    print(" Live Camera Running!")
    print(" Controls:")
    print("   [Q] or [ESC] : Exit")
    print("   [S]          : Save Snapshot")
    print("=" * 60 + "\n")

    snapshot_counter = 0
    fps_smooth = 30.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame from camera. Exiting...")
                break

            # Send frame to C++ inference engine (zero-copy memory access)
            t0 = time.perf_counter()
            detections, metrics = detector.detect(frame)
            loop_time = (time.perf_counter() - t0) * 1000.0

            if metrics.total_ms > 0:
                fps_smooth = fps_smooth * 0.9 + metrics.fps * 0.1

            # Render UI overlays using Python
            annotated_frame = detector.draw(frame, detections, metrics=metrics, show_hud=True)

            # Display window
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                snapshot_counter += 1
                filename = f"snapshot_{snapshot_counter}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"[Snapshot Saved] {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("\nCamera closed successfully.")

if __name__ == "__main__":
    main()
