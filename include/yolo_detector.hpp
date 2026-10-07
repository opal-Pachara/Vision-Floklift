#pragma once

#include "types.hpp"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <array>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace yolo {

class YoloDetector {
public:
    explicit YoloDetector(const ModelConfig& config = ModelConfig());
    ~YoloDetector() = default;

    // Initialize session and warm up
    void init();
    void warmup(int iterations = 3);

#ifdef HAVE_OPENCV
    // Primary detection method with OpenCV Mat (BGR format)
    std::vector<Detection> detect(const cv::Mat& image);

    // Draw bounding boxes, labels, and performance HUD on image
    void draw_detections(cv::Mat& image, 
                         const std::vector<Detection>& detections, 
                         bool show_hud = true);
#endif

    // Overload for raw RGB/BGR byte buffer (works without OpenCV)
    std::vector<Detection> detect_raw(const uint8_t* image_data, 
                                      int width, 
                                      int height, 
                                      int channels, 
                                      bool is_bgr = true);

    const PerformanceMetrics& get_last_metrics() const { return last_metrics_; }
    const std::vector<std::string>& get_class_names() const { return class_names_; }
    const ModelConfig& get_config() const { return config_; }

    // Dynamic threshold update without re-loading model
    void set_thresholds(const std::vector<float>& class_thresholds, float nms_threshold);

private:
    void load_labels();
    LetterboxInfo compute_letterbox(int src_w, int src_h, int target_w, int target_h);

    // Preprocessing: BGR/RGB to Letterbox CHW float buffer normalized to [0, 1]
    void preprocess(const uint8_t* src_data, int src_w, int src_h, int channels, bool is_bgr);

    // Postprocessing: Decode candidate boxes [1, 8, 8400] and apply NMS
    std::vector<Detection> postprocess(const float* output_data, 
                                      const std::vector<int64_t>& output_shape, 
                                      int orig_w, 
                                      int orig_h);

    // Fast IoU computation for Non-Maximum Suppression
    static float compute_iou(const BoundingBox& a, const BoundingBox& b);
    std::vector<int> nms(const std::vector<BoundingBox>& boxes, 
                         const std::vector<float>& scores, 
                         float threshold);

private:
    ModelConfig config_;
    PerformanceMetrics last_metrics_;
    std::vector<std::string> class_names_;
    LetterboxInfo current_letterbox_;

    // ONNX Runtime session components
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;

    // Input/output tensor specs
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;
    std::vector<const char*> input_node_name_ptrs_;
    std::vector<const char*> output_node_name_ptrs_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;

    // Pre-allocated continuous memory for input tensor to avoid heap allocations per frame
    std::vector<float> input_tensor_values_;
};

} // namespace yolo
