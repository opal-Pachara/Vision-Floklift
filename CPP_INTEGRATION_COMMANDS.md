# คู่มือคำสั่งปฏิบัติการ: การประยุกต์ C++ เชื่อมต่อเข้ากับระบบอื่น (C++ Integration Commands Cheatsheet)

เอกสารนี้รวบรวม **คำสั่ง Terminal และโค้ดตัวอย่างจริง (Executable Commands & Recipes)** สำหรับการนำ C++ Inference Engine ไปประยุกต์ใช้งานร่วมกับ 5 ระบบหลักในงานอุตสาหกรรมและโลจิสติกส์:

---

## 📑 สารบัญระบบที่นำไปเชื่อมต่อ
1. [ระบบที่ 1: เชื่อม C++ เข้ากับ Web API (FastAPI AI Microservice)](#ระบบที่-1-เชื่อม-c-เข้ากับ-web-api-fastapi-ai-microservice)
2. [ระบบที่ 2: เชื่อม C++ กับกล้องวงจรปิดโกดัง (RTSP IP Camera Stream)](#ระบบที่-2-เชื่อม-c-กับกล้องวงจรปิดโกดัง-rtsp-ip-camera-stream)
3. [ระบบที่ 3: เชื่อม C++ เข้ากับระบบหุ่นยนต์โกดัง (ROS2 / Robot Operating System)](#ระบบที่-3-เชื่อม-c-เข้ากับระบบหุ่นยนต์โกดัง-ros2)
4. [ระบบที่ 4: เชื่อม C++ เข้ากับ IoT & Smart Dashboard (MQTT Protocol)](#ระบบที่-4-เชื่อม-c-เข้ากับ-iot--smart-dashboard-mqtt-protocol)
5. [ระบบที่ 5: การบรรจุลง Docker Container สำหรับ Production](#ระบบที่-5-การบรรจุลง-docker-container-สำหรับ-production)
6. [ชุดคำสั่ง Prompt สำหรับใช้สั่ง AI เรียนรู้ต่อยอด (AI Learning Prompts)](#ชุดคำสั่ง-prompt-สำหรับใช้สั่ง-ai-เรียนรู้ต่อยอด)

---

## ระบบที่ 1: เชื่อม C++ เข้ากับ Web API (FastAPI AI Microservice)

สร้าง REST API Server เพื่อให้เว็บ, แอปมือถือ หรือระบบ ERP ส่งรูปมาตรวจจับ แล้ว C++ คำนวณส่งผลลัพธ์เป็น JSON กลับไปในเสี้ยววินาที

### 💻 คำสั่งติดตั้งเครื่องมือ:
```bash
pip install fastapi uvicorn python-multipart requests
```

### 📝 ไฟล์โค้ด: `server_api.py`
```python
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from yolo_cpp import YoloCppDetector

app = FastAPI(title="Logistics Vision C++ API")

# โหลดโมเดลผ่าน C++ Engine ครั้งเดียวตอนเริ่ม Server
detector = YoloCppDetector("best.onnx", "labels.txt", use_coreml=True)

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    # 1. อ่านไฟล์รูปภาพจาก Request เข้า NumPy
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. ส่งเข้า C++ Engine คำนวณ (Zero-Copy)
    detections, metrics = detector.detect(img)

    # 3. จัด Format เป็น JSON Response
    results = []
    for d in detections:
        results.append({
            "class_name": d.class_name,
            "confidence": round(d.confidence, 4),
            "bbox": {"x": d.x, "y": d.y, "width": d.width, "height": d.height}
        })

    return {
        "status": "success",
        "latency_ms": round(metrics.total_ms, 2),
        "fps": round(metrics.fps, 1),
        "total_detections": len(results),
        "detections": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 🚀 คำสั่งรัน Server และทดสอบยิง Request:
```bash
# 1. รัน Server
python3 server_api.py

# 2. ทดสอบยิง Request ด้วย cURL (เปิด Terminal อีกหน้าต่าง)
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_test.jpg"
```

---

## ระบบที่ 2: เชื่อม C++ กับกล้องวงจรปิดโกดัง (RTSP IP Camera Stream)

กล้องวงจรปิด (IP Camera) ส่งภาพผ่านโปรโตคอล `rtsp://` ปัญหาหลักคือการเกิด **Buffer Lag** (ภาพดีเลย์สะสม) หากประมวลผลไม่ทัน การใช้ C++ จะช่วยดึงภาพแบบ Real-Time ไม่สะสม Latency

### 📝 ไฟล์โค้ด: `rtsp_streamer.py`
```python
import cv2
import threading
from yolo_cpp import YoloCppDetector

# RTSP URL ตัวอย่าง (หรือ URL กล้องจริงของโกดัง)
RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream1"

class RTSPStreamReader:
    """อ่านภาพแยก Thread ป้องกันปัญหา Buffer ค้าง"""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# รันร่วมกับ C++ Detector
detector = YoloCppDetector("best.onnx", "labels.txt", use_coreml=True)
stream = RTSPStreamReader(0) # ใส่ RTSP_URL หรือเลขกล้อง 0

print("กำลังอ่านสตรีม... กด 'q' เพื่อออก")
while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    # C++ ทำนายผลแบบ Real-time
    detections, metrics = detector.detect(frame)
    annotated = detector.draw(frame, detections, metrics)

    cv2.imshow("Warehouse CCTV", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
```

---

## ระบบที่ 3: เชื่อม C++ เข้ากับระบบหุ่นยนต์โกดัง (ROS2)

ในระบบหุ่นยนต์ AGV หรือ Forklift ไร้คนขับ ระบบจะสื่อสารกันผ่าน **ROS2 (Robot Operating System)** ผ่าน Message Topic

### 💻 คำสั่งสร้าง ROS2 Package:
```bash
# 1. สร้าง Workspace และ Package
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python yolo_vision_ros

# 2. คัดลอก libyolo_detector.dylib และ yolo_cpp.py ไปไว้ใน package
```

### 📝 ไฟล์โค้ด ROS2 Node: `yolo_node.py`
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from yolo_cpp import YoloCppDetector
import json

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()
        self.detector = YoloCppDetector("best.onnx", "labels.txt")
        
        # Subscribe ภาพจากกล้องหุ่นยนต์
        self.subscription = self.create_subscription(
            Image, '/robot/camera/image_raw', self.image_callback, 10
        )
        
        # Publish ข้อมูลตรวจจับให้ระบบควบคุมพวงมาลัย/เบรก
        self.publisher = self.create_publisher(String, '/robot/vision/detections', 10)
        self.get_logger().info('C++ YOLO ROS2 Node พร้อมทำงานแล้ว!')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detections, metrics = self.detector.detect(frame)
        
        # ส่งแจ้งเตือนถ้ามีคนตัดหน้าหุ่นยนต์
        data = [{"class": d.class_name, "conf": d.confidence, "x": d.x, "y": d.y} for d in detections]
        msg_out = String()
        msg_out.data = json.dumps(data)
        self.publisher.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

## ระบบที่ 4: เชื่อม C++ เข้ากับ IoT & Smart Dashboard (MQTT Protocol)

ส่งสถานะความปลอดภัยและการตรวจนับสต็อกขึ้น Cloud / Dashboard เช่น Grafana, ThingsBoard หรือ Node-RED ผ่าน MQTT

### 💻 คำสั่งติดตั้ง:
```bash
pip install paho-mqtt
```

### 📝 ตัวอย่างโค้ด: `mqtt_safety_alert.py`
```python
import paho.mqtt.client as mqtt
import json
import time
import cv2
from yolo_cpp import YoloCppDetector

# เชื่อมต่อ MQTT Broker
client = mqtt.Client()
client.connect("broker.hivemq.com", 1883, 60)

detector = YoloCppDetector("best.onnx", "labels.txt", use_coreml=True)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    detections, metrics = detector.detect(frame)

    # นับจำนวน
    persons = sum(1 for d in detections if d.class_id == 2)
    helmets = sum(1 for d in detections if d.class_id == 1)
    forklifts = sum(1 for d in detections if d.class_id == 0)

    # ส่ง Telemetry ไปยัง Dashboard
    payload = {
        "timestamp": time.time(),
        "forklifts": forklifts,
        "persons": persons,
        "helmets": helmets,
        "safety_violation": (persons > helmets) # แจ้งเตือนคนไม่ใส่หมวก
    }
    
    client.publish("warehouse/zone_a/safety", json.dumps(payload))
    time.sleep(0.1) # ส่งข้อมูลทุก 100ms
```

---

## ระบบที่ 5: การบรรจุลง Docker Container สำหรับ Production

ช่วยให้โปรเจกต์รันได้บน Server Linux ทุกตัว ไม่ต้องกังวลเรื่องการตั้งค่าเครื่องหรือลง Library ซ้ำซ้อน

### 📝 ไฟล์ `Dockerfile`:
```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ติดตั้ง C++ Build Tools & OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libopencv-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอกโปรเจกต์
COPY . /app

# ติดตั้ง ONNX Runtime C++ และ Python Dependencies
RUN pip3 install --no-cache-dir numpy opencv-python fastapi uvicorn
RUN ./build.sh

EXPOSE 8000
CMD ["python3", "server_api.py"]
```

### 💻 คำสั่ง Build และ Run Docker:
```bash
# 1. Build Image
docker build -t vision-logistic:latest .

# 2. Run Container พร้อมเปิด Port 8000
docker run -d -p 8000:8000 --name logistic_vision vision-logistic:latest
```

---

## ชุดคำสั่ง Prompt สำหรับใช้สั่ง AI เรียนรู้ต่อยอด

เมื่อคุณต้องการทำโปรเจกต์ใหม่ และต้องการให้ AI ช่วยต่อยอดการประยุกต์ C++ เข้ากับระบบต่าง ๆ สามารถคัดลอก **Prompts สำเร็จรูป** เหล่านี้ไปถาม AI ได้ทันที:

### 🎯 Prompt 1: ขอให้แปลงฟังก์ชัน C++ เป็น Python Binding
> *"ผมมีฟังก์ชันประมวลผลภาพใน C++ ชื่อ `process_tracking()` รับ pointer ภาพ `uint8_t*` และพิกัด Bounding Box ช่วยเขียน `extern \"C\"` bridge และโค้ด Python `ctypes` แบบ Zero-Copy ให้หน่อย"*

### 🎯 Prompt 2: ขอให้นำโมเดล C++ ไปทำ Multi-Camera Pipeline
> *"ช่วยออกแบบระบบ C++ ที่อ่านภาพจากกล้อง RTSP พร้อมกัน 4 ตัว (Multi-Threading) โดยแชร์ Model Instance เดียวกัน เพื่อให้ได้ throughput รวมสูงสุด ไม่กิน RAM"*

### 🎯 Prompt 3: ขอให้นำระบบไปต่อกับฐานข้อมูล SQL/MongoDB
> *"เขียนโค้ด Python เชื่อมกับ `yolo_cpp.py` เพื่อบันทึกประวัติการตรวจพบรถ Forklift และการละเมิดความปลอดภัย (ไม่สวมหมวก/เสื้อกั๊ก) ลงฐานข้อมูล PostgreSQL พร้อมเวลา Timestamp"*
