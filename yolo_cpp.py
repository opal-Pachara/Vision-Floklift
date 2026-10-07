"""
YOLO C++ Inference Engine - Python Bridge
Enables high-performance C++ inference (CoreML/CPU/CUDA) from Python applications.
"""

import os
import sys
import ctypes
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

# C API Structures
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

class CDetectionMetrics(ctypes.Structure):
    _fields_ = [
        ("preprocess_ms", ctypes.c_double),
        ("inference_ms", ctypes.c_double),
        ("postprocess_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("fps", ctypes.c_double),
    ]

@dataclass
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def x1(self) -> int:
        return int(self.x)

    @property
    def y1(self) -> int:
        return int(self.y)

    @property
    def x2(self) -> int:
        return int(self.x + self.width)

    @property
    def y2(self) -> int:
        return int(self.y + self.height)

@dataclass
class PerformanceMetrics:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    fps: float

class YoloCppDetector:
    """
    Python wrapper around high-performance C++ YOLO Inference Engine
    """
    def __init__(
        self,
        model_path: str = "best.onnx",
        labels_path: str = "labels.txt",
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45,
        num_threads: int = 4,
        use_coreml: bool = True
    ):
        self._lib = self._load_library()
        self._setup_bindings()

        # Initialize detector instance in C++
        self._handle = self._lib.yolo_create(
            model_path.encode("utf-8"),
            labels_path.encode("utf-8"),
            ctypes.c_float(conf_thresh),
            ctypes.c_float(nms_thresh),
            ctypes.c_int(num_threads),
            ctypes.c_int(1 if use_coreml else 0)
        )

        if not self._handle:
            raise RuntimeError(f"Failed to initialize C++ YOLO Detector with model: {model_path}")

        # Pre-allocate buffer for detections
        self._max_detections = 200
        self._c_detections = (CDetection * self._max_detections)()
        self._c_metrics = CDetectionMetrics()

        # Distinct colors for each class
        self.class_colors = {
            0: (0, 140, 255),  # Forklift: Orange
            1: (0, 255, 255),  # Helmet: Yellow
            2: (255, 178, 50), # Person: Blue/Cyan
            3: (50, 255, 120), # Safety Vest: Green
        }

    def _load_library(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(curr_dir, "build", "libyolo_detector.dylib"),
            os.path.join(curr_dir, "build", "libyolo_detector.so"),
            os.path.join(curr_dir, "libyolo_detector.dylib"),
            os.path.join(curr_dir, "libyolo_detector.so"),
            os.path.join(curr_dir, "build", "Release", "yolo_detector.dll"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return ctypes.CDLL(path)

        raise FileNotFoundError(
            "C++ shared library 'libyolo_detector' not found! "
            "Please run './build.sh' to compile the C++ library first."
        )

    def _setup_bindings(self):
        # yolo_create
        self._lib.yolo_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.yolo_create.restype = ctypes.c_void_p

        # yolo_detect
        self._lib.yolo_detect.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p, # uint8_t* pointer
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(CDetection),
            ctypes.c_int,
            ctypes.POINTER(CDetectionMetrics),
        ]
        self._lib.yolo_detect.restype = ctypes.c_int

        # yolo_set_class_thresholds
        self._lib.yolo_set_class_thresholds.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float
        ]
        self._lib.yolo_set_class_thresholds.restype = None

        # yolo_free
        self._lib.yolo_free.argtypes = [ctypes.c_void_p]
        self._lib.yolo_free.restype = None

    def set_thresholds(self, class_thresholds: List[float], nms_threshold: float = 0.45):
        """
        Dynamically update per-class thresholds and NMS IoU threshold in C++
        class_thresholds: list of 4 floats [forklift, helmet, person, safety_vest]
        """
        c_arr = (ctypes.c_float * len(class_thresholds))(*class_thresholds)
        self._lib.yolo_set_class_thresholds(
            self._handle,
            c_arr,
            ctypes.c_int(len(class_thresholds)),
            ctypes.c_float(nms_threshold)
        )

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], PerformanceMetrics]:
        """
        Run inference using C++ engine with zero-copy numpy memory access.
        image: BGR numpy uint8 image array (from cv2.imread or cv2.VideoCapture)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Expected numpy ndarray for image")

        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        height, width, channels = image.shape
        data_ptr = image.ctypes.data_as(ctypes.c_char_p)

        # Call C++ engine
        num_dets = self._lib.yolo_detect(
            self._handle,
            data_ptr,
            ctypes.c_int(width),
            ctypes.c_int(height),
            ctypes.c_int(channels),
            ctypes.c_int(1), # is_bgr = True
            self._c_detections,
            ctypes.c_int(self._max_detections),
            ctypes.byref(self._c_metrics)
        )

        detections = []
        for i in range(num_dets):
            cdet = self._c_detections[i]
            detections.append(Detection(
                x=float(cdet.x),
                y=float(cdet.y),
                width=float(cdet.width),
                height=float(cdet.height),
                confidence=float(cdet.confidence),
                class_id=int(cdet.class_id),
                class_name=cdet.class_name.decode("utf-8")
            ))

        metrics = PerformanceMetrics(
            preprocess_ms=float(self._c_metrics.preprocess_ms),
            inference_ms=float(self._c_metrics.inference_ms),
            postprocess_ms=float(self._c_metrics.postprocess_ms),
            total_ms=float(self._c_metrics.total_ms),
            fps=float(self._c_metrics.fps),
        )

        return detections, metrics

    def draw(
        self, 
        image: np.ndarray, 
        detections: List[Detection], 
        metrics: Optional[PerformanceMetrics] = None,
        show_hud: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and HUD overlay onto the image.
        """
        annotated = image.copy()
        counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for det in detections:
            counts[det.class_id] = counts.get(det.class_id, 0) + 1
            color = self.class_colors.get(det.class_id, (0, 255, 0))

            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = f"{det.class_name} {det.confidence * 100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            top = max(y1, th + 6)
            cv2.rectangle(annotated, (x1, top - th - 6), (x1 + tw + 8, top), color, -1)
            cv2.putText(annotated, label, (x1 + 4, top - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        if show_hud:
            h, w = annotated.shape[:2]
            # Top dark status banner
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

            # Performance text
            if metrics:
                perf_txt = f"{metrics.fps:.1f} FPS | C++ Engine: {metrics.total_ms:.1f} ms"
            else:
                perf_txt = "C++ Inference Engine"
            cv2.putText(annotated, perf_txt, (15, 24), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Object counts
            cnt_txt = f"Forklift: {counts.get(0, 0)}  |  Person: {counts.get(2, 0)}  |  Helmet: {counts.get(1, 0)}  |  Vest: {counts.get(3, 0)}"
            cv2.putText(annotated, cnt_txt, (15, 47), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Safety compliance alert
            persons = counts.get(2, 0)
            helmets = counts.get(1, 0)
            vests = counts.get(3, 0)
            if persons > 0 and (helmets < persons or vests < persons):
                cv2.putText(annotated, "SAFETY ALERT: Missing PPE!", (w - 330, 36), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 70, 255), 2, cv2.LINE_AA)
            elif persons > 0:
                cv2.putText(annotated, "ALL PPE COMPLIANT", (w - 270, 36), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50), 2, cv2.LINE_AA)

        return annotated

    def close(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.yolo_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
