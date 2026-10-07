#include "yolo_detector.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif

namespace yolo {

YoloDetector::YoloDetector(const ModelConfig& config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_WARNING, "YoloDetector"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    init();
}

void YoloDetector::init() {
    load_labels();

    // Configure ONNX Runtime session for maximum speed
    session_options_.SetIntraOpNumThreads(config_.num_threads);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

#if defined(__APPLE__)
    if (config_.use_coreml) {
        try {
            // Attempt to enable CoreML execution provider on macOS Apple Silicon
            uint32_t coreml_flags = 0;
            auto status = OrtSessionOptionsAppendExecutionProvider_CoreML(static_cast<OrtSessionOptions*>(session_options_), coreml_flags);
            if (status != nullptr) {
                std::cout << "[YoloDetector] CoreML unavailable, using optimized CPU." << std::endl;
            } else {
                std::cout << "[YoloDetector] Apple CoreML Execution Provider enabled." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "[YoloDetector] CoreML unavailable, falling back to CPU: " << e.what() << std::endl;
        }
    }
#endif

#if defined(USE_CUDA)
    if (config_.use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config_.device_id;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[YoloDetector] CUDA Execution Provider enabled on device " << config_.device_id << std::endl;
        } catch (const std::exception& e) {
            std::cout << "[YoloDetector] CUDA unavailable, falling back to CPU: " << e.what() << std::endl;
        }
    }
#endif

    // Create session
    std::cout << "[YoloDetector] Loading ONNX model: " << config_.model_path << std::endl;
    session_ = std::make_unique<Ort::Session>(env_, config_.model_path.c_str(), session_options_);

    // Inspect Model Inputs
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names_.clear();
    input_node_name_ptrs_.clear();

    for (size_t i = 0; i < num_input_nodes; ++i) {
        auto input_name = session_->GetInputNameAllocated(i, allocator);
        input_node_names_.push_back(input_name.get());
    }
    for (const auto& name : input_node_names_) {
        input_node_name_ptrs_.push_back(name.c_str());
    }

    auto input_type_info = session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = input_tensor_info.GetShape();

    // Default to [1, 3, 640, 640] if dynamic or unspecified
    if (input_shape_.size() == 4) {
        if (input_shape_[0] <= 0) input_shape_[0] = 1;
        if (input_shape_[2] <= 0) input_shape_[2] = config_.input_height;
        if (input_shape_[3] <= 0) input_shape_[3] = config_.input_width;
    } else {
        input_shape_ = {1, 3, config_.input_height, config_.input_width};
    }

    // Inspect Model Outputs
    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names_.clear();
    output_node_name_ptrs_.clear();

    for (size_t i = 0; i < num_output_nodes; ++i) {
        auto output_name = session_->GetOutputNameAllocated(i, allocator);
        output_node_names_.push_back(output_name.get());
    }
    for (const auto& name : output_node_names_) {
        output_node_name_ptrs_.push_back(name.c_str());
    }

    auto output_type_info = session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = output_tensor_info.GetShape();

    // Pre-allocate contiguous input memory buffer to avoid malloc during runtime
    size_t total_input_elements = input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3];
    input_tensor_values_.resize(total_input_elements, 0.0f);

    std::cout << "[YoloDetector] Initialized successfully!" << std::endl;
    std::cout << "  Input: " << input_node_names_[0] << " [" 
              << input_shape_[0] << "x" << input_shape_[1] << "x" 
              << input_shape_[2] << "x" << input_shape_[3] << "]" << std::endl;
    std::cout << "  Output: " << output_node_names_[0] << " [";
    for (size_t i = 0; i < output_shape_.size(); ++i) {
        std::cout << output_shape_[i] << (i + 1 < output_shape_.size() ? "x" : "");
    }
    std::cout << "]" << std::endl;
    std::cout << "  Classes: " << class_names_.size() << std::endl;
}

void YoloDetector::load_labels() {
    class_names_.clear();
    std::ifstream file(config_.labels_path);
    if (!file.is_open()) {
        std::cout << "[YoloDetector] Notice: labels file '" << config_.labels_path 
                  << "' not found. Using default indices." << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        // Strip trailing CR/LF
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (!line.empty()) {
            class_names_.push_back(line);
        }
    }
}

LetterboxInfo YoloDetector::compute_letterbox(int src_w, int src_h, int target_w, int target_h) {
    LetterboxInfo info;
    float scale = std::min(static_cast<float>(target_w) / src_w, static_cast<float>(target_h) / src_h);
    int new_unpad_w = static_cast<int>(std::round(src_w * scale));
    int new_unpad_h = static_cast<int>(std::round(src_h * scale));

    float pad_w = (target_w - new_unpad_w) * 0.5f;
    float pad_h = (target_h - new_unpad_h) * 0.5f;

    info.scale = scale;
    info.pad_w = pad_w;
    info.pad_h = pad_h;
    info.new_unpad_w = new_unpad_w;
    info.new_unpad_h = new_unpad_h;
    return info;
}

#ifdef HAVE_OPENCV
std::vector<Detection> YoloDetector::detect(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    return detect_raw(image.data, image.cols, image.rows, image.channels(), true);
}
#endif

std::vector<Detection> YoloDetector::detect_raw(const uint8_t* image_data, 
                                                int width, 
                                                int height, 
                                                int channels, 
                                                bool is_bgr) {
    if (!image_data || width <= 0 || height <= 0 || channels < 3) {
        return {};
    }

    auto t_start_total = std::chrono::high_resolution_clock::now();

    // 1. Preprocessing
    auto t_start_pre = std::chrono::high_resolution_clock::now();
    current_letterbox_ = compute_letterbox(width, height, config_.input_width, config_.input_height);
    preprocess(image_data, width, height, channels, is_bgr);
    auto t_end_pre = std::chrono::high_resolution_clock::now();

    // 2. Inference
    auto t_start_inf = std::chrono::high_resolution_clock::now();
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        input_tensor_values_.data(),
        input_tensor_values_.size(),
        input_shape_.data(),
        input_shape_.size()
    );

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_name_ptrs_.data(),
        &input_tensor,
        1,
        output_node_name_ptrs_.data(),
        output_node_name_ptrs_.size()
    );
    auto t_end_inf = std::chrono::high_resolution_clock::now();

    // 3. Postprocessing
    auto t_start_post = std::chrono::high_resolution_clock::now();
    const float* output_data = output_tensors[0].GetTensorData<float>();
    auto actual_output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::vector<Detection> detections = postprocess(output_data, actual_output_shape, width, height);
    auto t_end_post = std::chrono::high_resolution_clock::now();

    // Update performance metrics
    last_metrics_.preprocess_time_ms = std::chrono::duration<double, std::milli>(t_end_pre - t_start_pre).count();
    last_metrics_.inference_time_ms = std::chrono::duration<double, std::milli>(t_end_inf - t_start_inf).count();
    last_metrics_.postprocess_time_ms = std::chrono::duration<double, std::milli>(t_end_post - t_start_post).count();
    last_metrics_.total_time_ms = std::chrono::duration<double, std::milli>(t_end_post - t_start_total).count();
    last_metrics_.fps = (last_metrics_.total_time_ms > 0.0) ? (1000.0 / last_metrics_.total_time_ms) : 0.0;

    return detections;
}

void YoloDetector::preprocess(const uint8_t* src_data, int src_w, int src_h, int channels, bool is_bgr) {
    const int target_w = config_.input_width;
    const int target_h = config_.input_height;
    const int target_area = target_w * target_h;

    // Fill buffer with letterbox padding color (114.0f / 255.0f = 0.4470588f)
    const float pad_val = 114.0f / 255.0f;
    std::fill(input_tensor_values_.begin(), input_tensor_values_.end(), pad_val);

    float* r_plane = input_tensor_values_.data();
    float* g_plane = r_plane + target_area;
    float* b_plane = g_plane + target_area;

    const int unpad_w = current_letterbox_.new_unpad_w;
    const int unpad_h = current_letterbox_.new_unpad_h;
    const int x_offset = static_cast<int>(std::round(current_letterbox_.pad_w));
    const int y_offset = static_cast<int>(std::round(current_letterbox_.pad_h));

    const float scale_x = static_cast<float>(src_w) / unpad_w;
    const float scale_y = static_cast<float>(src_h) / unpad_h;
    const float inv_255 = 1.0f / 255.0f;

    // Fast bilinear sampling directly into planar CHW memory
    for (int dy = 0; dy < unpad_h; ++dy) {
        float sy = (dy + 0.5f) * scale_y - 0.5f;
        int sy0 = std::max(0, static_cast<int>(std::floor(sy)));
        int sy1 = std::min(src_h - 1, sy0 + 1);
        float fy = sy - sy0;
        float inv_fy = 1.0f - fy;

        int canvas_y = y_offset + dy;
        if (canvas_y < 0 || canvas_y >= target_h) continue;

        for (int dx = 0; dx < unpad_w; ++dx) {
            int canvas_x = x_offset + dx;
            if (canvas_x < 0 || canvas_x >= target_w) continue;

            float sx = (dx + 0.5f) * scale_x - 0.5f;
            int sx0 = std::max(0, static_cast<int>(std::floor(sx)));
            int sx1 = std::min(src_w - 1, sx0 + 1);
            float fx = sx - sx0;
            float inv_fx = 1.0f - fx;

            float w00 = inv_fx * inv_fy;
            float w01 = fx * inv_fy;
            float w10 = inv_fx * fy;
            float w11 = fx * fy;

            const uint8_t* p00 = src_data + (sy0 * src_w + sx0) * channels;
            const uint8_t* p01 = src_data + (sy0 * src_w + sx1) * channels;
            const uint8_t* p10 = src_data + (sy1 * src_w + sx0) * channels;
            const uint8_t* p11 = src_data + (sy1 * src_w + sx1) * channels;

            float c0 = p00[0] * w00 + p01[0] * w01 + p10[0] * w10 + p11[0] * w11;
            float c1 = p00[1] * w00 + p01[1] * w01 + p10[1] * w10 + p11[1] * w11;
            float c2 = p00[2] * w00 + p01[2] * w01 + p10[2] * w10 + p11[2] * w11;

            int dst_idx = canvas_y * target_w + canvas_x;
            if (is_bgr) {
                // Input is BGR: c0=B, c1=G, c2=R -> Output CHW order is RGB
                r_plane[dst_idx] = c2 * inv_255;
                g_plane[dst_idx] = c1 * inv_255;
                b_plane[dst_idx] = c0 * inv_255;
            } else {
                // Input is RGB
                r_plane[dst_idx] = c0 * inv_255;
                g_plane[dst_idx] = c1 * inv_255;
                b_plane[dst_idx] = c2 * inv_255;
            }
        }
    }
}

std::vector<Detection> YoloDetector::postprocess(const float* output_data, 
                                                const std::vector<int64_t>& shape, 
                                                int orig_w, 
                                                int orig_h) {
    if (!output_data || shape.size() < 3) {
        return {};
    }

    // Determine tensor dimension layout:
    // Format A: [1, 8, 8400] -> channels=8, num_anchors=8400
    // Format B: [1, 8400, 8] -> num_anchors=8400, channels=8
    int num_channels = 0;
    int num_anchors = 0;
    bool transposed = false;

    if (shape[1] < shape[2]) {
        // [1, 8, 8400]
        num_channels = static_cast<int>(shape[1]);
        num_anchors = static_cast<int>(shape[2]);
        transposed = false;
    } else {
        // [1, 8400, 8]
        num_anchors = static_cast<int>(shape[1]);
        num_channels = static_cast<int>(shape[2]);
        transposed = true;
    }

    int num_classes = num_channels - 4;
    if (num_classes <= 0) return {};

    const float conf_thresh = config_.conf_threshold;
    const float pad_w = current_letterbox_.pad_w;
    const float pad_h = current_letterbox_.pad_h;
    const float inv_scale = 1.0f / current_letterbox_.scale;

    std::vector<BoundingBox> candidate_boxes;
    std::vector<float> candidate_scores;
    std::vector<int> candidate_classes;
    candidate_boxes.reserve(256);
    candidate_scores.reserve(256);
    candidate_classes.reserve(256);

    for (int i = 0; i < num_anchors; ++i) {
        // Find best class score
        float max_score = 0.0f;
        int best_class_id = -1;

        if (!transposed) {
            // Layout [1, num_channels, num_anchors]
            for (int c = 0; c < num_classes; ++c) {
                float score = output_data[(4 + c) * num_anchors + i];
                if (score > max_score) {
                    max_score = score;
                    best_class_id = c;
                }
            }
        } else {
            // Layout [1, num_anchors, num_channels]
            const float* row = output_data + i * num_channels;
            for (int c = 0; c < num_classes; ++c) {
                float score = row[4 + c];
                if (score > max_score) {
                    max_score = score;
                    best_class_id = c;
                }
            }
        }

        float threshold_for_class = conf_thresh;
        if (best_class_id >= 0 && best_class_id < static_cast<int>(config_.class_thresholds.size())) {
            threshold_for_class = config_.class_thresholds[best_class_id];
        }

        if (max_score < threshold_for_class) {
            continue;
        }

        // Decode cx, cy, w, h
        float cx, cy, w, h;
        if (!transposed) {
            cx = output_data[0 * num_anchors + i];
            cy = output_data[1 * num_anchors + i];
            w  = output_data[2 * num_anchors + i];
            h  = output_data[3 * num_anchors + i];
        } else {
            const float* row = output_data + i * num_channels;
            cx = row[0];
            cy = row[1];
            w  = row[2];
            h  = row[3];
        }

        // Map back to original image space
        float x0 = (cx - w * 0.5f - pad_w) * inv_scale;
        float y0 = (cy - h * 0.5f - pad_h) * inv_scale;
        float box_w = w * inv_scale;
        float box_h = h * inv_scale;

        // Clip to image boundary
        x0 = std::max(0.0f, std::min(x0, static_cast<float>(orig_w - 1)));
        y0 = std::max(0.0f, std::min(y0, static_cast<float>(orig_h - 1)));
        box_w = std::max(1.0f, std::min(box_w, static_cast<float>(orig_w) - x0));
        box_h = std::max(1.0f, std::min(box_h, static_cast<float>(orig_h) - y0));

        BoundingBox bbox;
        bbox.x = x0;
        bbox.y = y0;
        bbox.width = box_w;
        bbox.height = box_h;

        candidate_boxes.push_back(bbox);
        candidate_scores.push_back(max_score);
        candidate_classes.push_back(best_class_id);
    }

    if (candidate_boxes.empty()) {
        return {};
    }

    // Apply Non-Maximum Suppression per class to prevent cross-class suppression
    std::vector<Detection> final_detections;
    for (int c = 0; c < num_classes; ++c) {
        std::vector<BoundingBox> class_boxes;
        std::vector<float> class_scores;
        std::vector<size_t> class_orig_indices;

        for (size_t idx = 0; idx < candidate_classes.size(); ++idx) {
            if (candidate_classes[idx] == c) {
                class_boxes.push_back(candidate_boxes[idx]);
                class_scores.push_back(candidate_scores[idx]);
                class_orig_indices.push_back(idx);
            }
        }

        if (class_boxes.empty()) continue;

        std::vector<int> keep_indices = nms(class_boxes, class_scores, config_.nms_threshold);
        for (int k_idx : keep_indices) {
            Detection det;
            det.bbox = class_boxes[k_idx];
            det.confidence = class_scores[k_idx];
            det.class_id = c;
            det.class_name = (c < static_cast<int>(class_names_.size())) 
                             ? class_names_[c] 
                             : ("class_" + std::to_string(c));
            final_detections.push_back(det);
        }
    }

    return final_detections;
}

float YoloDetector::compute_iou(const BoundingBox& a, const BoundingBox& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.right(), b.right());
    float y2 = std::min(a.bottom(), b.bottom());

    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;

    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

std::vector<int> YoloDetector::nms(const std::vector<BoundingBox>& boxes, 
                                   const std::vector<float>& scores, 
                                   float threshold) {
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);

    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        keep.push_back(idx);

        for (size_t j = i + 1; j < indices.size(); ++j) {
            int next_idx = indices[j];
            if (suppressed[next_idx]) continue;

            if (compute_iou(boxes[idx], boxes[next_idx]) > threshold) {
                suppressed[next_idx] = true;
            }
        }
    }
    return keep;
}

void YoloDetector::set_thresholds(const std::vector<float>& class_thresholds, float nms_threshold) {
    if (!class_thresholds.empty()) {
        config_.class_thresholds = class_thresholds;
    }
    if (nms_threshold > 0.0f) {
        config_.nms_threshold = nms_threshold;
    }
}

void YoloDetector::warmup(int iterations) {
    std::cout << "[YoloDetector] Warming up session with " << iterations << " iterations..." << std::endl;
    std::vector<uint8_t> dummy_img(config_.input_width * config_.input_height * 3, 128);
    for (int i = 0; i < iterations; ++i) {
        detect_raw(dummy_img.data(), config_.input_width, config_.input_height, 3, true);
    }
    std::cout << "[YoloDetector] Warmup complete!" << std::endl;
}

#ifdef HAVE_OPENCV
void YoloDetector::draw_detections(cv::Mat& image, 
                                   const std::vector<Detection>& detections, 
                                   bool show_hud) {
    // Distinct colors for each class: forklift, helmet, person, safety_vest
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0, 165, 255), // Forklift: Orange
        cv::Scalar(0, 255, 255), // Helmet: Yellow
        cv::Scalar(255, 100, 0), // Person: Blue
        cv::Scalar(0, 255, 0)    // Safety Vest: Green
    };

    for (const auto& det : detections) {
        cv::Scalar color = palette[det.class_id % palette.size()];

        cv::Rect rect(static_cast<int>(det.bbox.x), 
                      static_cast<int>(det.bbox.y), 
                      static_cast<int>(det.bbox.width), 
                      static_cast<int>(det.bbox.height));

        // Draw bounding box
        cv::rectangle(image, rect, color, 2, cv::LINE_AA);

        // Format label text
        char label_buf[128];
        snprintf(label_buf, sizeof(label_buf), "%s %.1f%%", det.class_name.c_str(), det.confidence * 100.0f);
        std::string label = label_buf;

        int base_line = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);

        int top = std::max(rect.y, label_size.height + 4);
        cv::Rect bg_rect(rect.x, top - label_size.height - 4, label_size.width + 6, label_size.height + 6);
        cv::rectangle(image, bg_rect, color, cv::FILLED);
        cv::putText(image, label, cv::Point(rect.x + 3, top - 2), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    if (show_hud) {
        char hud_buf[256];
        snprintf(hud_buf, sizeof(hud_buf), 
                 "Total: %.1fms (Inf: %.1fms | Pre: %.1fms | Post: %.1fms) | %.1f FPS", 
                 last_metrics_.total_time_ms,
                 last_metrics_.inference_time_ms,
                 last_metrics_.preprocess_time_ms,
                 last_metrics_.postprocess_time_ms,
                 last_metrics_.fps);

        cv::putText(image, hud_buf, cv::Point(15, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(image, hud_buf, cv::Point(15, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
}
#endif

} // namespace yolo
