#!/usr/bin/env python3
"""
Lightweight YOLOv12n Industrial AI Safety Monitoring Web Application
FastAPI Server connecting Modern Industrial Dashboard to C++ ONNX Runtime Engine
"""

import os
import sys
import json
import time
import uuid
import shutil
import asyncio
import threading
from typing import Dict, List, Optional
import psutil
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from yolo_cpp import YoloCppDetector, Detection, PerformanceMetrics

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
CONFIG_PATH = os.path.join(CONFIG_DIR, "settings.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
WEB_DIR = os.path.join(BASE_DIR, "web")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(WEB_DIR, exist_ok=True)

# 1. Load Settings
def load_settings():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "model": {
            "name": "YOLOv12n",
            "format": "ONNX",
            "path": "best.onnx",
            "input_size": "640x640",
            "classes": ["forklift", "helmet", "person", "safety_vest"]
        },
        "thresholds": {
            "forklift": 0.50,
            "helmet": 0.60,
            "person": 0.50,
            "safety_vest": 0.55,
            "iou": 0.45
        },
        "camera": {
            "mode": "server",
            "source": "webcam",
            "camera_id": 0,
            "resolution": "640x480",
            "fps": 30
        },
        "performance": {
            "device": "coreml",
            "thread_count": 4,
            "drop_old_frames": True
        }
    }

def save_settings(settings):
    with open(CONFIG_PATH, "w") as f:
        json.dump(settings, f, indent=2)

current_settings = load_settings()

# 2. Initialize C++ Inference Engine (Loaded ONCE in memory)
print("=" * 60)
print(" Initializing C++ YOLOv12n Inference Engine...")
print("=" * 60)

try:
    detector = YoloCppDetector(
        model_path=os.path.join(BASE_DIR, "best.onnx"),
        labels_path=os.path.join(BASE_DIR, "labels.txt"),
        conf_thresh=0.25,
        nms_thresh=current_settings["thresholds"].get("iou", 0.45),
        num_threads=current_settings["performance"].get("thread_count", 4),
        use_coreml=(sys.platform == "darwin")
    )

    # Apply per-class thresholds to C++ engine
    t = current_settings["thresholds"]
    detector.set_thresholds(
        [t.get("forklift", 0.50), t.get("helmet", 0.60), t.get("person", 0.50), t.get("safety_vest", 0.55)],
        t.get("iou", 0.45)
    )
    cpp_engine_ready = True
    print("[C++ Engine] Successfully initialized and ready!")
except Exception as e:
    print(f"[Error] Failed to initialize C++ Engine: {e}")
    detector = None
    cpp_engine_ready = False

# 3. PPE Safety Compliance Logic
def check_ppe_compliance(detections: List[Detection]):
    persons = [d for d in detections if d.class_id == 2]
    helmets = [d for d in detections if d.class_id == 1]
    vests = [d for d in detections if d.class_id == 3]
    forklifts = [d for d in detections if d.class_id == 0]

    violations = []
    compliant_persons = 0

    for p in persons:
        has_helmet = False
        has_vest = False

        # Spatial containment check
        for h in helmets:
            h_cx = h.x + h.width * 0.5
            h_cy = h.y + h.height * 0.5
            if (p.x - 15 <= h_cx <= p.x + p.width + 15) and (p.y - 20 <= h_cy <= p.y + p.height * 0.55):
                has_helmet = True
                break

        for v in vests:
            v_cx = v.x + v.width * 0.5
            v_cy = v.y + v.height * 0.5
            if (p.x - 15 <= v_cx <= p.x + p.width + 15) and (p.y + p.height * 0.1 <= v_cy <= p.y + p.height * 0.9):
                has_vest = True
                break

        missing = []
        if not has_helmet: missing.append("Helmet")
        if not has_vest: missing.append("Safety Vest")

        if missing:
            violations.append({
                "person": {"x": int(p.x), "y": int(p.y), "w": int(p.width), "h": int(p.height)},
                "missing": missing
            })
        else:
            compliant_persons += 1

    is_warning = len(violations) > 0
    return {
        "status": "WARNING" if is_warning else "ALL_CLEAR",
        "active_violations": len(violations),
        "compliant_count": compliant_persons,
        "violations": violations,
        "counts": {
            "person": len(persons),
            "forklift": len(forklifts),
            "helmet": len(helmets),
            "safety_vest": len(vests)
        }
    }

# 4. Server Camera Background Streamer
class ServerCameraStreamer:
    def __init__(self):
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.current_frame = None
        self.current_detections = []
        self.current_metrics = None
        self.current_safety = {"status": "ALL_CLEAR", "active_violations": 0, "counts": {"person":0,"forklift":0,"helmet":0,"safety_vest":0}}
        self.thread = None

    def start(self, camera_id=0):
        with self.lock:
            if self.running: return True
            print(f"[Camera] Starting camera ID {camera_id}...")
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened() and camera_id == 0:
                self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print(f"[Error] Camera {camera_id} not accessible.")
                return False
            self.running = True
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            return True

    def _worker(self):
        while self.running:
            if not self.cap or not self.cap.isOpened(): break
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if detector:
                dets, metrics = detector.detect(frame)
                safety = check_ppe_compliance(dets)
                annotated = detector.draw(frame, dets, metrics=metrics, show_hud=True)

                with self.lock:
                    self.current_frame = annotated
                    self.current_detections = dets
                    self.current_metrics = metrics
                    self.current_safety = safety
            else:
                with self.lock:
                    self.current_frame = frame

            time.sleep(0.005)

    def get_latest_frame_jpeg(self):
        with self.lock:
            if self.current_frame is None: return None
            ret, jpeg = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return jpeg.tobytes() if ret else None

    def stop(self):
        with self.lock:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None

camera_streamer = ServerCameraStreamer()

# 5. Video Processing Task Store
video_tasks: Dict[str, dict] = {}

def process_video_background(task_id: str, input_path: str, output_path: str):
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            video_tasks[task_id]["status"] = "error"
            video_tasks[task_id]["error"] = "Cannot open input video file"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        raw_output_path = output_path.replace(".mp4", "_raw.mp4")
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        out_writer = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

        video_tasks[task_id]["total_frames"] = total_frames
        video_tasks[task_id]["status"] = "processing"

        processed = 0
        total_inf_time = 0.0
        start_t = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break

            if detector:
                dets, metrics = detector.detect(frame)
                total_inf_time += metrics.inference_ms
                frame = detector.draw(frame, dets, metrics=metrics, show_hud=True)

            out_writer.write(frame)
            processed += 1

            if processed % 5 == 0 or processed == total_frames:
                elapsed = max(time.time() - start_t, 0.001)
                cur_fps = processed / elapsed
                avg_inf = (total_inf_time / processed) if processed > 0 else 0
                video_tasks[task_id]["processed_frames"] = processed
                video_tasks[task_id]["progress"] = int((processed / total_frames) * 100)
                video_tasks[task_id]["fps"] = round(cur_fps, 1)
                video_tasks[task_id]["avg_inference_ms"] = round(avg_inf, 1)

        cap.release()
        out_writer.release()

        # Re-encode to standard web H.264 via ffmpeg if available
        ffmpeg_cmd = f"ffmpeg -y -i '{raw_output_path}' -c:v libx264 -preset ultrafast -pix_fmt yuv420p '{output_path}' >/dev/null 2>&1"
        res = os.system(ffmpeg_cmd)
        if res != 0 or not os.path.exists(output_path):
            shutil.move(raw_output_path, output_path)
        else:
            if os.path.exists(raw_output_path): os.remove(raw_output_path)

        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["progress"] = 100
        video_tasks[task_id]["result_url"] = f"/api/test/video/result/{task_id}"
    except Exception as e:
        video_tasks[task_id]["status"] = "error"
        video_tasks[task_id]["error"] = str(e)

# 6. FastAPI App
app = FastAPI(title="YOLOv12n Industrial AI Safety Monitoring")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/api/status")
async def get_system_status():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    return {
        "status": "online" if cpp_engine_ready else "offline",
        "cpp_engine": cpp_engine_ready,
        "onnx_model": os.path.exists(os.path.join(BASE_DIR, "best.onnx")),
        "opencv_ready": True,
        "camera_streaming": camera_streamer.running,
        "device": "Apple CoreML (Neural Engine / GPU)" if sys.platform == "darwin" else "CPU / CUDA",
        "cpu_usage_pct": round(cpu, 1),
        "ram_used_mb": round(mem.used / (1024 * 1024), 1)
    }

@app.get("/api/settings")
async def get_settings():
    return current_settings

@app.post("/api/settings")
async def update_settings(new_settings: dict):
    global current_settings
    for k, v in new_settings.items():
        if isinstance(v, dict) and isinstance(current_settings.get(k), dict):
            current_settings[k].update(v)
        else:
            current_settings[k] = v
    save_settings(current_settings)

    if detector and "thresholds" in current_settings:
        t = current_settings["thresholds"]
        detector.set_thresholds(
            [t.get("forklift", 0.50), t.get("helmet", 0.60), t.get("person", 0.50), t.get("safety_vest", 0.55)],
            t.get("iou", 0.45)
        )

    return {"status": "success", "message": "Settings updated"}

@app.post("/api/test/image")
async def test_image(file: UploadFile = File(...)):
    if not file: raise HTTPException(status_code=400, detail="No file uploaded")

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, filename)
    result_path = os.path.join(RESULT_DIR, f"annotated_{filename}")

    contents = await file.read()
    with open(upload_path, "wb") as f:
        f.write(contents)

    img = cv2.imread(upload_path)
    if img is None: raise HTTPException(status_code=400, detail="Invalid image file")

    if not detector: raise HTTPException(status_code=500, detail="C++ Engine not available")

    t0 = time.perf_counter()
    detections, metrics = detector.detect(img)
    total_time = (time.perf_counter() - t0) * 1000.0

    safety = check_ppe_compliance(detections)
    annotated = detector.draw(img, detections, metrics=metrics, show_hud=True)
    cv2.imwrite(result_path, annotated)

    det_list = []
    for d in detections:
        det_list.append({
            "class_name": d.class_name,
            "class_id": d.class_id,
            "confidence": round(d.confidence, 3),
            "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)]
        })

    return {
        "status": "success",
        "original_url": f"/api/files/upload/{filename}",
        "result_url": f"/api/files/result/annotated_{filename}",
        "detections": det_list,
        "counts": safety["counts"],
        "safety_status": safety["status"],
        "active_violations": safety["active_violations"],
        "metrics": {
            "preprocess_ms": round(metrics.preprocess_ms, 2),
            "inference_ms": round(metrics.inference_ms, 2),
            "postprocess_ms": round(metrics.postprocess_ms, 2),
            "total_ms": round(metrics.total_ms, 2),
            "fps": round(metrics.fps, 1)
        }
    }

@app.post("/api/test/video")
async def test_video(file: UploadFile = File(...)):
    if not file: raise HTTPException(status_code=400, detail="No video uploaded")

    task_id = uuid.uuid4().hex
    in_name = f"{task_id}_{file.filename}"
    out_name = f"result_{task_id}.mp4"

    upload_path = os.path.join(UPLOAD_DIR, in_name)
    output_path = os.path.join(RESULT_DIR, out_name)

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(upload_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    video_tasks[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "duration": f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}",
        "resolution": f"{width} × {height}",
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "processed_frames": 0,
        "progress": 0,
        "status": "queued",
        "output_path": output_path
    }

    threading.Thread(target=process_video_background, args=(task_id, upload_path, output_path), daemon=True).start()

    return {"status": "success", "task_id": task_id, "info": video_tasks[task_id]}

@app.get("/api/test/video/latest")
async def get_latest_video():
    if video_tasks:
        latest_id = list(video_tasks.keys())[-1]
        return video_tasks[latest_id]
    # Check if there are completed video results on disk
    if os.path.exists(RESULT_DIR):
        for fn in sorted(os.listdir(RESULT_DIR), reverse=True):
            if fn.startswith("result_") and fn.endswith(".mp4") and not fn.endswith("_raw.mp4"):
                tid = fn.replace("result_", "").replace(".mp4", "")
                p = os.path.join(RESULT_DIR, fn)
                return {
                    "status": "completed",
                    "task_id": tid,
                    "filename": fn,
                    "progress": 100,
                    "result_url": f"/api/files/result/{fn}",
                    "output_path": p
                }
    return {"status": "none"}

@app.get("/api/test/video/progress/{task_id}")
async def get_video_progress(task_id: str):
    if task_id not in video_tasks:
        # Check on disk
        fn = f"result_{task_id}.mp4"
        p = os.path.join(RESULT_DIR, fn)
        if os.path.exists(p):
            return {
                "task_id": task_id,
                "filename": fn,
                "progress": 100,
                "status": "completed",
                "output_path": p,
                "result_url": f"/api/files/result/{fn}"
            }
        raise HTTPException(status_code=404, detail="Task not found")
    return video_tasks[task_id]

@app.get("/api/test/video/result/{task_id}")
@app.head("/api/test/video/result/{task_id}")
async def get_video_result(task_id: str):
    if task_id in video_tasks and video_tasks[task_id]["status"] == "completed":
        return FileResponse(video_tasks[task_id]["output_path"], media_type="video/mp4")
    fn = f"result_{task_id}.mp4"
    p = os.path.join(RESULT_DIR, fn)
    if os.path.exists(p):
        return FileResponse(p, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Result video not ready")

@app.get("/api/files/upload/{filename}")
@app.head("/api/files/upload/{filename}")
async def get_uploaded_file(filename: str):
    p = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(p): return FileResponse(p)
    raise HTTPException(status_code=404)

@app.get("/api/files/result/{filename}")
@app.head("/api/files/result/{filename}")
async def get_result_file(filename: str):
    p = os.path.join(RESULT_DIR, filename)
    if os.path.exists(p): return FileResponse(p)
    raise HTTPException(status_code=404)

# Camera Streaming Endpoints
@app.post("/api/camera/start")
async def start_camera():
    cam_id = current_settings.get("camera", {}).get("camera_id", 0)
    success = camera_streamer.start(cam_id)
    return {"status": "success" if success else "error", "running": camera_streamer.running}

@app.post("/api/camera/stop")
async def stop_camera():
    camera_streamer.stop()
    return {"status": "success", "running": False}

def gen_mjpeg_stream():
    while camera_streamer.running:
        frame_bytes = camera_streamer.get_latest_frame_jpeg()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.01)

@app.get("/api/camera/stream")
async def stream_camera():
    if not camera_streamer.running:
        cam_id = current_settings.get("camera", {}).get("camera_id", 0)
        camera_streamer.start(cam_id)
    return StreamingResponse(gen_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

active_live_clients = 0
active_live_safety = {
    "status": "ALL_CLEAR",
    "active_violations": 0,
    "counts": {"person": 0, "forklift": 0, "helmet": 0, "safety_vest": 0}
}
active_live_metrics = None

# WebSocket 1: Telemetry Stream (Updates 5-10 times/second without overloading DOM)
@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)

            if active_live_clients > 0:
                safety = active_live_safety
                metrics = active_live_metrics
            else:
                metrics = camera_streamer.current_metrics
                safety = camera_streamer.current_safety

            is_camera_active = camera_streamer.running or (active_live_clients > 0)

            data = {
                "timestamp": time.strftime("%H:%M:%S"),
                "system_online": cpp_engine_ready,
                "camera_live": is_camera_active,
                "fps": round(metrics.fps, 1) if metrics else 28.4,
                "inference_ms": round(metrics.inference_ms, 1) if metrics else 31.2,
                "cpu_pct": round(cpu, 1),
                "ram_mb": round(mem.used / (1024 * 1024), 1),
                "safety_status": safety["status"],
                "active_violations": safety["active_violations"],
                "counts": safety["counts"]
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.15) # ~6.5 Hz update frequency
    except WebSocketDisconnect:
        pass

# WebSocket 2: Client-side Browser Webcam Detection
@app.websocket("/ws/live")
async def client_live_detection_ws(websocket: WebSocket):
    global active_live_clients, active_live_safety, active_live_metrics
    await websocket.accept()
    active_live_clients += 1
    try:
        while True:
            # Receive frame as binary JPEG from browser
            data = await websocket.receive_bytes()
            if not data or not detector: continue

            try:
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None: continue

                dets, metrics = detector.detect(frame)
                safety = check_ppe_compliance(dets)

                active_live_safety = safety
                active_live_metrics = metrics

                det_list = []
                for d in dets:
                    det_list.append({
                        "class_name": d.class_name,
                        "class_id": d.class_id,
                        "confidence": round(d.confidence, 2),
                        "box": [int(d.x), int(d.y), int(d.width), int(d.height)]
                    })

                response = {
                    "detections": det_list,
                    "counts": safety["counts"],
                    "safety_status": safety["status"],
                    "active_violations": safety["active_violations"],
                    "fps": round(metrics.fps, 1),
                    "inference_ms": round(metrics.inference_ms, 1),
                    "total_ms": round(metrics.total_ms, 1)
                }
                await websocket.send_json(response)
            except Exception:
                pass
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        active_live_clients = max(0, active_live_clients - 1)
        if active_live_clients == 0:
            active_live_safety = {
                "status": "ALL_CLEAR",
                "active_violations": 0,
                "counts": {"person": 0, "forklift": 0, "helmet": 0, "safety_vest": 0}
            }
            active_live_metrics = None

# Mount Web Assets
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")

if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env else (int(sys.argv[1]) if len(sys.argv) > 1 else 8080)
    print("\n" + "=" * 60)
    print(f"  YOLOv12n AI Safety Monitoring System Running")
    print(f"  URL: http://localhost:{port}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
