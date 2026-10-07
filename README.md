# YOLO C++ High-Performance Inference Engine (vision-logistic)

ระบบ C++ Inference Engine ประสิทธิภาพสูงสำหรับการรันโมเดล **YOLOv12n / YOLOv11 / YOLOv8 (`best.pt`)** เพื่อเพิ่มความเร็ว (FPS) และลด Latency ในงาน Computer Vision และงานโลจิสติกส์ (ตรวจจับ `forklift`, `helmet`, `person`, `safety_vest`)

---

## 🚀 ทำไม C++ ถึงเร็วกว่า Python PyTorch?

| ขั้นตอน | Python (Ultralytics PyTorch) | C++ Inference Engine (ONNX Runtime) | การ Optimize ในโค้ด C++ |
| :--- | :--- | :--- | :--- |
| **Runtime Overhead** | ติด GIL, Python Interpreter, PyTorch dynamic dispatch | คอมไพล์เป็น Native Machine Code (-O3, ARM NEON / AVX) | ลด Overhead ของภาษาลงเหลือศูนย์ |
| **Memory Allocation** | จอง memory ใหม่ทุก frame (malloc/free) | **Pre-allocated Tensor Buffers** | จองบัฟเฟอร์ Input/Output ครั้งเดียวตอน Initialize ไม่มีการ allocate ซ้ำระหว่าง frame |
| **Letterbox Preprocessing** | Resize + Pad + Transpose HWC->CHW ผ่านหลาย copy | **Direct Single-Pass Bilinear Interpolation** | แปลงภาพ, normalize และจัดเรียง planar CHW ลง tensor buffer ในลูปเดียว |
| **Inference Engine** | PyTorch CPU/MPS | **ONNX Runtime + Apple CoreML / CUDA / Accelerate** | ใช้ Apple Neural Engine (ANE) & GPU บน Mac หรือ TensorRT/CUDA บน Linux |
| **Postprocessing & NMS** | Python loop + Tensor indexing | **Vectorized Anchor Filtering + Cache-friendly NMS** | กรอง Anchor ที่ confidence ต่ำทิ้งทันที และคำนวณ IoU เฉพาะคลาส (0.03 - 0.06 ms) |

---

## 📊 ผลการทดสอบความเร็ว (Benchmark Results บน Apple Silicon)

โมเดล: `best.onnx` (YOLOv12n, 640x640, 4 Classes)

| โหมดการทำงาน | Preprocess Latency | Inference Latency | Postprocess Latency | Total Latency | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **C++ Apple CoreML (Neural Engine / GPU)** | ~4.3 ms | **~36.6 ms** | **0.03 ms** | **~41.0 ms** | **~24.4 FPS** |
| **C++ CPU Multi-threaded (4 Threads)** | ~3.8 ms | ~84.6 ms | 0.06 ms | ~88.5 ms | ~11.3 FPS |
| *Python PyTorch CPU (Baseline)* | ~15-25 ms | ~120-180 ms | ~15-30 ms | ~180-230 ms | ~4.5 - 5.5 FPS |

> **สรุป**: C++ กับ CoreML เร็วกว่า Python PyTorch ถึง **4 - 5 เท่า** และใช้ทรัพยากร CPU น้อยลงมาก!

---

## 📁 โครงสร้างโปรเจกต์

```text
vision-logistic/
├── best.pt                  # PyTorch model weights ดั้งเดิม
├── best.onnx                # โมเดล ONNX (Opset 18, 640x640) ที่ optimize แล้ว
├── labels.txt               # รายชื่อคลาส (forklift, helmet, person, safety_vest)
├── export_model.py          # สคริปต์ Python สำหรับแปลง .pt -> .onnx
├── build.sh                 # สคริปต์คอมไพล์โปรเจกต์ C++ แบบ 1 คลิก
├── benchmark.sh             # สคริปต์ทดสอบความเร็ว เปรียบเทียบ CoreML vs CPU
├── CMakeLists.txt           # Build configuration (ตรวจจับ ONNX Runtime และ OpenCV อัตโนมัติ)
├── include/
│   ├── types.hpp            # โครงสร้างข้อมูล Detection, BoundingBox, ModelConfig, Metrics
│   ├── yolo_detector.hpp    # Class Interface สำหรับ YOLO C++ Detector
│   ├── stb_image.h          # Header โหลดภาพแบบ Zero-dependency
│   └── stb_image_write.h    # Header บันทึกภาพแบบ Zero-dependency
└── src/
    ├── yolo_detector.cpp    # ตัวขับเคลื่อนหลัก (Session, Preprocess, Inference, Postprocess, NMS)
    └── main.cpp             # CLI Application (รองรับ Image, Video, Webcam, Benchmark)
```

---

## 🛠️ การติดตั้ง Dependencies

### บน macOS (Apple Silicon / Intel):
```bash
brew install cmake onnxruntime opencv
```

### บน Ubuntu / Debian Linux:
```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
# ดาวน์โหลด ONNX Runtime C++ release:
# https://github.com/microsoft/onnxruntime/releases
```

---

## ⚙️ ขั้นตอนการ Build และ Compile

เพียงรันคำสั่ง:
```bash
./build.sh
```

หรือใช้ CMake แบบ Manual:
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

โปรแกรมจะสร้างไฟล์ Execute ชื่อ `./build/yolo_cpp`

---

## 💻 วิธีการรันใช้งาน (Commands)

### 1. ทดสอบความเร็ว (Benchmark Latency & FPS)
```bash
# รัน Benchmark เปรียบเทียบทั้ง CoreML และ CPU (50 รอบ)
./benchmark.sh 50

# หรือระบุโหมดเอง:
./build/yolo_cpp --coreml --benchmark 100
./build/yolo_cpp --cpu --threads 4 --benchmark 100
```

### 2. รัน Inference กับรูปภาพเดี่ยว (Image Detection)
```bash
./build/yolo_cpp --image sample_test.jpg --output output.jpg
```
โปรแกรมจะแสดง Bounding Box, คลาส, ความมั่นใจ (Confidence %) พร้อมเวลา Latency และบันทึกรูปผลลัพธ์ไปที่ `output.jpg`

### 3. รัน Inference กับวิดีโอ หรือกล้อง Webcam แบบ Real-time
```bash
# วิธีที่ 1: รันผ่าน Python UI + C++ Engine (Hybrid - ยืดหยุ่นสูงสุด):
python3 run_camera.py        # ใช้กล้องเริ่มต้น (ID 0)
python3 run_camera.py 1      # กล้อง USB (ID 1)
# -> Python จัดการกล้อง/UI และส่งภาพเข้า C++ Engine เพื่อรัน CoreML แบบ Zero-Copy

# วิธีที่ 2: รันผ่านโปรแกรม C++ ล้วน:
./run_camera.sh              # กล้อง ID 0
./build/yolo_cpp --video 0   # ผ่าน CLI หลัก

# วิธีที่ 3: รันไฟล์วิดีโอ:
./build/yolo_cpp --video warehouse_cctv.mp4 --output result_video.mp4
```

### 4. ปรับจูนพารามิเตอร์ (Options)
```text
  --conf <float>       ปรับค่า Confidence Threshold (default: 0.25)
  --nms <float>        ปรับค่า NMS IoU Threshold (default: 0.45)
  --threads <int>      กำหนดจำนวน Thread สำหรับ CPU (default: 4)
  --coreml             เปิดใช้งาน Apple Silicon CoreML (default บน macOS)
  --cpu                บังคับรันบน CPU
```

---

## 🔄 วิธีการ Export โมเดลใหม่จาก `.pt`

หากมีการ Train โมเดลใหม่ในอนาคต สามารถ Export เป็น ONNX ได้ง่ายๆ:
```bash
python3 export_model.py --weights best.pt --imgsz 640 --opset 18
```
สคริปต์จะสร้าง `best.onnx` และอัปเดต `labels.txt` ให้ตรงกับโมเดลโดยอัตโนมัติ

---

## 🧩 การนำ `yolo_detector` ไป Integrate ในโปรเจกต์ C++ อื่น

สามารถ `#include "yolo_detector.hpp"` แล้วนำคลาส `yolo::YoloDetector` ไปใช้งานได้ทันที:

```cpp
#include "yolo_detector.hpp"

yolo::ModelConfig config;
config.model_path = "best.onnx";
config.labels_path = "labels.txt";
config.conf_threshold = 0.30f;
config.use_coreml = true;

yolo::YoloDetector detector(config);

// ส่ง cv::Mat เข้าไป
cv::Mat frame = cv::imread("forklift.jpg");
std::vector<yolo::Detection> detections = detector.detect(frame);

for (const auto& det : detections) {
    std::cout << det.class_name << ": " << det.confidence 
              << " at [" << det.bbox.x << ", " << det.bbox.y << "]\n";
}

// วาดกล่องพร้อม HUD
detector.draw_detections(frame, detections);
cv::imwrite("result.jpg", frame);
```
