# คู่มือการเรียนรู้: การพัฒนา C++ ร่วมกับ Python แบบข้ามแพลตฟอร์ม (Cross-Platform C++/Python Architecture)

> **เป้าหมายของเอกสารนี้:**  
> เพื่อเป็นคู่มือศึกษาทีละขั้นตอน (Step-by-Step Guide) สำหรับผู้ที่ต้องการเรียนรู้การพัฒนาโมเดล AI และระบบ Computer Vision ประสิทธิภาพสูง โดยใช้ **C++ เป็นเครื่องยนต์ประมวลผล (High Performance Engine)** และเชื่อมต่อกับ **Python เป็นส่วนติดต่อผู้ใช้ (High Productivity UI/App)** ให้สามารถนำไปคอมไพล์และรันได้บนทุกระบบปฏิบัติการ (**macOS, Linux/Jetson, Windows**)

---

## 📑 สารบัญ
1. [ทำไมต้องใช้สถาปัตยกรรม Hybrid (C++ + Python)?](#1-ทำไมต้องใช้สถาปัตยกรรม-hybrid-c--python)
2. [เข้าใจความต่างของแต่ละระบบปฏิบัติการ (OS Differences)](#2-เข้าใจความต่างของแต่ละระบบปฏิบัติการ-os-differences)
3. [หลักการสำคัญ 4 เสาหลัก (4 Core Pillars)](#3-หลักการสำคัญ-4-เสาหลัก-4-core-pillars)
4. [เจาะลึกทีละขั้นตอน (Step-by-Step Implementation)](#4-เจาะลึกทีละขั้นตอน-step-by-step-implementation)
   - [Step 1: การออกแบบ C API Boundary (include/yolo_c_api.h)](#step-1-การออกแบบ-c-api-boundary)
   - [Step 2: การเขียนโค้ด C++ Bridge (src/yolo_c_api.cpp)](#step-2-การเขียนโค้ด-c-bridge)
   - [Step 3: การเขียน CMakeLists.txt ให้รองรับทุก OS](#step-3-การเขียน-cmakeliststxt-ให้รองรับทุก-os)
   - [Step 4: เทคนิค Zero-Copy ด้วย Python ctypes (yolo_cpp.py)](#step-4-เทคนิค-zero-copy-ด้วย-python-ctypes)
5. [การเร่งความเร็วฮาร์ดแวร์ข้ามแพลตฟอร์ม (Hardware Acceleration)](#5-การเร่งความเร็วฮาร์ดแวร์ข้ามแพลตฟอร์ม-hardware-acceleration)
6. [คำสั่ง Build ในแต่ละระบบปฏิบัติการ](#6-คำสั่ง-build-ในแต่ละระบบปฏิบัติการ)
7. [แบบฝึกหัดพัฒนาต่อยอด (Hands-On Exercises)](#7-แบบฝึกหัดพัฒนาต่อยอด-hands-on-exercises)

---

## 1. ทำไมต้องใช้สถาปัตยกรรม Hybrid (C++ + Python)?

ในงาน Computer Vision และโลจิสติกส์ระดับอุตสาหกรรม:
* **ถ้าเขียน Python ล้วน**: พัฒนาง่าย แต่ติดปัญหา **GIL (Global Interpreter Lock)**, Python Interpreter Overhead และ Garbage Collector ทำให้ FPS ตกและกิน CPU สูง
* **ถ้าเขียน C++ ล้วน**: เร็วสูงสุด แต่การทำ GUI, การต่อ Web Framework (FastAPI/Flask), การเขียนเชื่อมฐานข้อมูล หรือ Dashboard ทำได้ช้าและซับซ้อน

### โซลูชันที่ดีที่สุด:
| หน้าที่ | ภาษาที่ใช้ | ประโยชน์ |
| :--- | :--- | :--- |
| **Heavy Math & Inference** | **C++** | ทำ Letterbox, คูณ Tensor, รัน GPU/NPU, คำนวณ NMS ภายใน **0.04 ms** |
| **Business Logic & UI** | **Python** | เปิดกล้อง, ตรวจสอบเงื่อนไขความปลอดภัย, แสดงผลหน้าจอ, ส่ง Alert เข้า LINE/Database |

---

## 2. เข้าใจความต่างของแต่ละระบบปฏิบัติการ (OS Differences)

เมื่อเราคอมไพล์โค้ด C++ เป็น **Shared Dynamic Library** เพื่อให้ Python โหลดไปใช้ แต่ละระบบปฏิบัติการจะมีนามสกุลและรูปแบบต่างกัน:

| ระบบปฏิบัติการ | ชนิดไฟล์ไบนารี | ตัวคอมไพเลอร์ยอดนิยม | Hardware Acceleration หลัก |
| :--- | :---: | :--- | :--- |
| **macOS** | `.dylib` (Dynamic Library) | AppleClang | **Apple CoreML** (Apple Neural Engine / GPU) |
| **Linux (Ubuntu / Jetson)** | `.so` (Shared Object) | GCC / Clang | **NVIDIA TensorRT / CUDA** |
| **Windows** | `.dll` (Dynamic Link Library) | MSVC (Visual Studio) | **DirectML / CUDA** |

---

## 3. หลักการสำคัญ 4 เสาหลัก (4 Core Pillars)

เพื่อให้โค้ดข้ามแพลตฟอร์มได้อย่างราบรื่น ต้องยึดหลักการ 4 ข้อนี้:

### เสาหลักที่ 1: `extern "C"` ป้องกัน Name Mangling
ภาษา C++ มีฟีเจอร์ Function Overloading ทำให้คอมไพเลอร์แอบเปลี่ยนชื่อฟังก์ชันในไบนารี เช่น `yolo_create` อาจกลายเป็น `_Z11yolo_createPcS_ffii` ทำให้ Python หาชื่อฟังก์ชันไม่เจอ  
**วิธีแก้:** ครอบหัวฟังก์ชันด้วย `extern "C"` เพื่อบังคับให้คอมไพเลอร์เก็บชื่อฟังก์ชันแบบภาษา C ดั้งเดิม

### เสาหลักที่ 2: ใช้ข้อมูลชนิด Plain Old Data (POD Types)
ที่จุดเชื่อมต่อระหว่าง C++ กับ Python **ห้ามใช้คลาสซับซ้อนของ C++ ข้ามภาษา** เช่น `std::string`, `std::vector`, หรือ `cv::Mat`  
**วิธีที่ถูกต้อง:** ใช้ Type พื้นฐานของภาษา C เท่านั้น เช่น `int`, `float`, `char*`, `unsigned char*` หรือ `struct` ธรรมดา

### เสาหลักที่ 3: Zero-Copy Memory Sharing
ไม่แปลงรูปภาพซ้ำซ้อนใน Python แต่ส่ง **Memory Address (Pointer)** ของ NumPy Array ตรงเข้า C++ ทำให้ประหยัดเวลา CPU ไป 100%

### เสาหลักที่ 4: Macro สำหรับ Export ฟังก์ชัน
Windows ต้องใช้คำสั่ง `__declspec(dllexport)` ขณะที่ Linux/macOS ใช้ `__attribute__((visibility("default")))`

---

## 4. เจาะลึกทีละขั้นตอน (Step-by-Step Implementation)

### Step 1: การออกแบบ C API Boundary
ไฟล์: `include/yolo_c_api.h`

```c
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 1. จัดการ Macro ข้าม OS (Windows vs Linux/Mac)
#ifdef _WIN32
  #define YOLO_API __declspec(dllexport)
#else
  #define YOLO_API __attribute__((visibility("default")))
#endif

// 2. Struct ที่ทั้ง C++ และ Python เข้าใจตรงกัน
struct CDetection {
    float x;
    float y;
    float width;
    float height;
    float confidence;
    int class_id;
    char class_name[64];
};

struct CDetectionMetrics {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
    double fps;
};

// 3. ฟังก์ชันเปิดให้ภายนอกเรียกใช้งาน
YOLO_API void* yolo_create(const char* model_path, const char* labels_path, float conf_thresh, float nms_thresh, int num_threads, int use_coreml);
YOLO_API int yolo_detect(void* handle, const unsigned char* image_data, int width, int height, int channels, int is_bgr, CDetection* out_detections, int max_detections, CDetectionMetrics* out_metrics);
YOLO_API void yolo_free(void* handle);

#ifdef __cplusplus
}
#endif
```

---

### Step 2: การเขียนโค้ด C++ Bridge
ไฟล์: `src/yolo_c_api.cpp`

```cpp
#include "yolo_c_api.h"
#include "yolo_detector.hpp"
#include <cstring>

extern "C" {

void* yolo_create(const char* model_path, const char* labels_path, float conf_thresh, float nms_thresh, int num_threads, int use_coreml) {
    try {
        yolo::ModelConfig config;
        config.model_path = model_path;
        config.labels_path = labels_path;
        config.conf_threshold = conf_thresh;
        config.nms_threshold = nms_thresh;
        config.num_threads = num_threads;
        config.use_coreml = (use_coreml != 0);

        // จอง memory บน Heap แล้วส่ง pointer กลับไปให้ Python ถือไว้
        return static_cast<void*>(new yolo::YoloDetector(config));
    } catch (...) {
        return nullptr;
    }
}

int yolo_detect(void* handle, const unsigned char* image_data, int width, int height, int channels, int is_bgr, CDetection* out_detections, int max_detections, CDetectionMetrics* out_metrics) {
    if (!handle || !image_data || !out_detections) return 0;

    auto* detector = static_cast<yolo::YoloDetector*>(handle);
    
    // รันการคำนวณใน C++ ล้วนๆ (Letterbox -> CoreML -> NMS)
    auto results = detector->detect_raw(image_data, width, height, channels, is_bgr != 0);

    // เขียนผลลัพธ์กลับลง Buffer ที่ Python จองไว้
    int count = std::min(static_cast<int>(results.size()), max_detections);
    for (int i = 0; i < count; ++i) {
        out_detections[i].x = results[i].bbox.x;
        out_detections[i].y = results[i].bbox.y;
        out_detections[i].width = results[i].bbox.width;
        out_detections[i].height = results[i].bbox.height;
        out_detections[i].confidence = results[i].confidence;
        out_detections[i].class_id = results[i].class_id;
        std::strncpy(out_detections[i].class_name, results[i].class_name.c_str(), 63);
    }
    return count;
}

void yolo_free(void* handle) {
    if (handle) {
        delete static_cast<yolo::YoloDetector*>(handle);
    }
}

}
```

---

### Step 3: การเขียน CMakeLists.txt ให้รองรับทุก OS

หัวใจของการ Cross-Platform ใน C++ คือ **CMake** ตัวอย่างการเขียนเพื่อสร้างไฟล์ Shared Library อัตโนมัติ:

```cmake
cmake_minimum_required(VERSION 3.16)
project(yolo_vision_logistic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. ปรับ Optimization Flags ตามคอมไพเลอร์ของแต่ละ OS
if(MSVC)
    # บน Windows (Visual Studio)
    add_compile_options(/O2 /Oi /Ot /fp:fast)
else()
    # บน macOS และ Linux
    add_compile_options(-O3 -fPIC -Wall)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
        # Apple Silicon หรือ Jetson ARM NEON
        message(STATUS "Enabling ARM NEON Optimizations")
    else()
        # Intel / AMD x86_64
        add_compile_options(-march=native)
    endif()
endif()

# 2. ค้นหา ONNX Runtime
find_package(PkgConfig QUIET)
# ... ค้นหา libonnxruntime ...

# 3. สั่งคอมไพล์เป็น SHARED LIBRARY
# CMake จะสร้าง .dylib บน Mac, .so บน Linux, .dll บน Windows ให้อัตโนมัติ!
add_library(yolo_detector_shared SHARED 
    src/yolo_detector.cpp 
    src/yolo_c_api.cpp
)
set_target_properties(yolo_detector_shared PROPERTIES OUTPUT_NAME "yolo_detector")

# ลิงก์ Library
target_link_libraries(yolo_detector_shared PUBLIC ${ORT_LIBRARIES})
if(APPLE)
    find_library(COREML_FRAMEWORK CoreML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(yolo_detector_shared PUBLIC ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()
```

---

### Step 4: เทคนิค Zero-Copy ด้วย Python ctypes
ไฟล์: `yolo_cpp.py`

```python
import os
import ctypes
import numpy as np

# 1. แมป C Struct ใน Python
class CDetection(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("width", ctypes.c_float),
        ("height", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("class_id", ctypes.c_int),
        ("class_name", ctypes.c_char * 64),
    ]

class YoloCppDetector:
    def __init__(self, model_path="best.onnx", labels_path="labels.txt"):
        # 2. ค้นหาและโหลดไฟล์ Library อัตโนมัติตาม OS
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(curr_dir, "build", "libyolo_detector.dylib"), # macOS
            os.path.join(curr_dir, "build", "libyolo_detector.so"),    # Linux
            os.path.join(curr_dir, "build", "Release", "yolo_detector.dll"), # Windows
        ]
        
        self.lib = None
        for path in candidates:
            if os.path.exists(path):
                self.lib = ctypes.CDLL(path)
                break
        
        if not self.lib:
            raise FileNotFoundError("ไม่พบไฟล์ C++ Shared Library!")

        # 3. ผูกฟังก์ชัน C++
        self.lib.yolo_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]
        self.lib.yolo_create.restype = ctypes.c_void_p
        
        self.handle = self.lib.yolo_create(
            model_path.encode(), labels_path.encode(), 0.25, 0.45, 4, 1
        )

        # จองบัฟเฟอร์รอรับผลลัพธ์
        self.out_dets = (CDetection * 200)()

    def detect(self, image: np.ndarray):
        # 4. เทคนิค Zero-Copy: ดึง Memory Address ของ NumPy Array
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        h, w, c = image.shape
        data_ptr = image.ctypes.data_as(ctypes.c_char_p)

        # เรียก C++ โดยตรง
        count = self.lib.yolo_detect(
            self.handle, data_ptr, w, h, c, 1, self.out_dets, 200, None
        )
        return [self.out_dets[i] for i in range(count)]
```

---

## 5. การเร่งความเร็วฮาร์ดแวร์ข้ามแพลตฟอร์ม (Hardware Acceleration)

ONNX Runtime รองรับ **Execution Providers (EP)** ซึ่งช่วยให้โมเดลรันบนฮาร์ดแวร์เฉพาะทางได้สูงสุด:

```text
[ YOLO ONNX Model ]
       │
       ├── บน macOS Apple Silicon ──► CoreML EP (Neural Engine / GPU) ──► ~30 FPS
       │
       ├── บน Linux / Jetson / Cloud ──► TensorRT / CUDA EP (NVIDIA GPU) ──► ~60-120 FPS
       │
       ├── บน Windows ──► DirectML EP (Intel/AMD/NVIDIA GPU)
       │
       └── CPU ทั่วไป ──► OpenMP + ARM NEON / AVX-512 (Vectorized CPU)
```

ในโค้ด C++ ของเรา ได้ใส่โค้ดเปิดใช้งานอัตโนมัติไว้แล้ว:
```cpp
#if defined(__APPLE__)
    // เปิด CoreML อัตโนมัติบน Mac
    OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, 0);
#elif defined(USE_CUDA)
    // เปิด CUDA อัตโนมัติบน Linux/Windows
    OrtCUDAProviderOptions cuda_opts;
    session_options.AppendExecutionProvider_CUDA(cuda_opts);
#endif
```

---

## 6. คำสั่ง Build ในแต่ละระบบปฏิบัติการ

### 🍏 บน macOS (Apple Silicon M1/M2/M3/M4):
```bash
# ติดตั้ง Tools ผ่าน Homebrew
brew install cmake onnxruntime opencv

# คอมไพล์
./build.sh
```

### 🐧 บน Linux (Ubuntu 22.04 / 24.04 หรือ NVIDIA Jetson):
```bash
# ติดตั้ง Tools
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev

# ดาวน์โหลด ONNX Runtime C++ จาก GitHub Releases แล้วรัน:
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 🪟 บน Windows (PowerShell + Visual Studio):
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

---

## 7. แบบฝึกหัดพัฒนาต่อยอด (Hands-On Exercises)

เพื่อฝึกทักษะให้เชี่ยวชาญ แนะนำให้ลองทำโจทย์เหล่านี้ตามลำดับ:

1. **โจทย์ที่ 1 (ปรับจูนพารามิเตอร์):**  
   ลองแก้ไขไฟล์ `run_camera.py` ให้รับค่า Confidence Threshold ผ่านการกดปุ่มบนคีย์บอร์ด (เช่น กด `+` เพิ่ม threshold, กด `-` ลด threshold)
2. **โจทย์ที่ 2 (เพิ่มฟังก์ชันคำนวณระยะห่าง):**  
   เพิ่มฟังก์ชันใน C++ เพื่อคำนวณระยะห่าง (Distance) ระหว่าง `forklift` กับ `person` หากเข้าใกล้กันเกินระยะที่กำหนด ให้ส่งสัญญาณเตือน Alert ทันที
3. **โจทย์ที่ 3 (ทำ Docker Container):**  
   เขียนไฟล์ `Dockerfile` รันบน Linux Ubuntu เพื่อให้ระบบสามารถ Deploy ขึ้น Server หรือคลาวด์ได้ด้วยคำสั่งเดียว
