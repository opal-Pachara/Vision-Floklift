#include "yolo_c_api.h"
#include "yolo_detector.hpp"
#include <cstring>
#include <algorithm>

extern "C" {

YOLO_EXPORT void* yolo_create(const char* model_path, 
                              const char* labels_path, 
                              float conf_thresh, 
                              float nms_thresh, 
                              int num_threads, 
                              int use_coreml) {
    try {
        yolo::ModelConfig config;
        if (model_path && strlen(model_path) > 0) {
            config.model_path = model_path;
        }
        if (labels_path && strlen(labels_path) > 0) {
            config.labels_path = labels_path;
        }
        config.conf_threshold = conf_thresh > 0.0f ? conf_thresh : 0.25f;
        config.nms_threshold = nms_thresh > 0.0f ? nms_thresh : 0.45f;
        config.num_threads = num_threads > 0 ? num_threads : 4;
        config.use_coreml = (use_coreml != 0);

        auto* detector = new yolo::YoloDetector(config);
        detector->warmup(2);
        return static_cast<void*>(detector);
    } catch (...) {
        return nullptr;
    }
}

YOLO_EXPORT int yolo_detect(void* handle, 
                            const unsigned char* image_data, 
                            int width, 
                            int height, 
                            int channels, 
                            int is_bgr, 
                            CDetection* out_detections, 
                            int max_detections, 
                            CDetectionMetrics* out_metrics) {
    if (!handle || !image_data || !out_detections || max_detections <= 0) {
        return 0;
    }

    auto* detector = static_cast<yolo::YoloDetector*>(handle);
    try {
        auto detections = detector->detect_raw(image_data, width, height, channels, is_bgr != 0);

        int count = std::min(static_cast<int>(detections.size()), max_detections);
        for (int i = 0; i < count; ++i) {
            out_detections[i].x = detections[i].bbox.x;
            out_detections[i].y = detections[i].bbox.y;
            out_detections[i].width = detections[i].bbox.width;
            out_detections[i].height = detections[i].bbox.height;
            out_detections[i].confidence = detections[i].confidence;
            out_detections[i].class_id = detections[i].class_id;

            // Copy class name safely
            std::strncpy(out_detections[i].class_name, 
                         detections[i].class_name.c_str(), 
                         sizeof(out_detections[i].class_name) - 1);
            out_detections[i].class_name[sizeof(out_detections[i].class_name) - 1] = '\0';
        }

        if (out_metrics) {
            const auto& m = detector->get_last_metrics();
            out_metrics->preprocess_ms = m.preprocess_time_ms;
            out_metrics->inference_ms = m.inference_time_ms;
            out_metrics->postprocess_ms = m.postprocess_time_ms;
            out_metrics->total_ms = m.total_time_ms;
            out_metrics->fps = m.fps;
        }

        return count;
    } catch (...) {
        return 0;
    }
}

YOLO_EXPORT void yolo_set_class_thresholds(void* handle, 
                                           const float* class_thresholds, 
                                           int num_classes, 
                                           float nms_thresh) {
    if (!handle || !class_thresholds || num_classes <= 0) {
        return;
    }
    auto* detector = static_cast<yolo::YoloDetector*>(handle);
    std::vector<float> thresh_vec(class_thresholds, class_thresholds + num_classes);
    detector->set_thresholds(thresh_vec, nms_thresh);
}

YOLO_EXPORT void yolo_free(void* handle) {
    if (handle) {
        auto* detector = static_cast<yolo::YoloDetector*>(handle);
        delete detector;
    }
}

} // extern "C"
