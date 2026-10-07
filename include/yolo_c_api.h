#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define YOLO_EXPORT __declspec(dllexport)
#else
#define YOLO_EXPORT __attribute__((visibility("default")))
#endif

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

// Create YOLO detector instance
YOLO_EXPORT void* yolo_create(const char* model_path, 
                              const char* labels_path, 
                              float conf_thresh, 
                              float nms_thresh, 
                              int num_threads, 
                              int use_coreml);

// Detect objects in raw image buffer (HWC uint8 BGR or RGB)
// Returns number of detected objects
YOLO_EXPORT int yolo_detect(void* handle, 
                            const unsigned char* image_data, 
                            int width, 
                            int height, 
                            int channels, 
                            int is_bgr, 
                            CDetection* out_detections, 
                            int max_detections, 
                            CDetectionMetrics* out_metrics);

// Dynamically update per-class confidence thresholds and IoU threshold
YOLO_EXPORT void yolo_set_class_thresholds(void* handle, 
                                           const float* class_thresholds, 
                                           int num_classes, 
                                           float nms_thresh);

// Destroy detector instance and free resources
YOLO_EXPORT void yolo_destroy(void* handle);

#ifdef __cplusplus
}
#endif
