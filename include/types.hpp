#pragma once

#include <string>
#include <vector>
#include <chrono>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace yolo {

struct BoundingBox {
    float x = 0.0f; // top-left x
    float y = 0.0f; // top-left y
    float width = 0.0f;
    float height = 0.0f;

    float right() const { return x + width; }
    float bottom() const { return y + height; }
};

struct Detection {
    BoundingBox bbox;
    float confidence = 0.0f;
    int class_id = -1;
    std::string class_name;
};

struct ModelConfig {
    std::string model_path = "best.onnx";
    std::string labels_path = "labels.txt";
    int input_width = 640;
    int input_height = 640;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    std::vector<float> class_thresholds = {0.50f, 0.60f, 0.50f, 0.55f}; // 0: forklift, 1: helmet, 2: person, 3: safety_vest
    int num_threads = 4;
    bool use_coreml = true; // CoreML execution provider on macOS Apple Silicon
    bool use_cuda = false;   // CUDA execution provider on NVIDIA GPUs
    int device_id = 0;
};

struct PerformanceMetrics {
    double preprocess_time_ms = 0.0;
    double inference_time_ms = 0.0;
    double postprocess_time_ms = 0.0;
    double total_time_ms = 0.0;
    double fps = 0.0;
};

struct LetterboxInfo {
    float scale = 1.0f;
    float pad_w = 0.0f;
    float pad_h = 0.0f;
    int new_unpad_w = 0;
    int new_unpad_h = 0;
};

} // namespace yolo
