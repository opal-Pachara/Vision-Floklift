#include "yolo_detector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>

int main(int argc, char** argv) {
    int camera_id = 0;
    if (argc > 1) {
        try {
            camera_id = std::stoi(argv[1]);
        } catch (...) {
            std::cerr << "Invalid camera ID: " << argv[1] << ", using default ID 0.\n";
            camera_id = 0;
        }
    }

    std::cout << "========================================================\n";
    std::cout << "  YOLO Logistics & Safety Live Camera Detection (C++)\n";
    std::cout << "========================================================\n";
    std::cout << "Starting camera (ID: " << camera_id << ")...\n";

    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cout << "Camera ID " << camera_id << " not found. Trying camera ID 1...\n";
        camera_id = 1;
        cap.open(camera_id);
    }

    if (!cap.isOpened()) {
        std::cerr << "\n[Error] Unable to access camera!\n";
        std::cerr << "Tip: Please ensure your camera permissions are enabled for this app in macOS System Settings > Privacy & Security > Camera.\n";
        return 1;
    }

    // Set camera resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    // Initialize YOLO detector
    yolo::ModelConfig config;
    config.model_path = "best.onnx";
    config.labels_path = "labels.txt";
    config.conf_threshold = 0.25f;
    config.nms_threshold = 0.45f;
    config.num_threads = 4;
    config.use_coreml = true; // Use Apple CoreML on macOS for maximum speed

    std::cout << "Loading model: " << config.model_path << " ...\n";
    std::unique_ptr<yolo::YoloDetector> detector;
    try {
        detector = std::make_unique<yolo::YoloDetector>(config);
        detector->warmup(3);
    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to load detector: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\nCamera ready! Controls:\n";
    std::cout << "  [Q] or [ESC] : Exit\n";
    std::cout << "  [S]          : Save current frame snapshot\n";
    std::cout << "  [+] / [-]    : Adjust confidence threshold (current: " << config.conf_threshold << ")\n";
    std::cout << "========================================================\n\n";

    const std::string window_name = "Logistics Safety Detection (YOLO C++)";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    // Color palette for classes:
    // 0: forklift (Orange), 1: helmet (Yellow), 2: person (Blue/Cyan), 3: safety_vest (Green)
    const std::vector<cv::Scalar> class_colors = {
        cv::Scalar(0, 140, 255), // forklift: Orange
        cv::Scalar(0, 255, 255), // helmet: Yellow
        cv::Scalar(255, 178, 50),// person: Sky Blue
        cv::Scalar(50, 255, 120) // safety_vest: Bright Green
    };

    cv::Mat frame;
    int snapshot_count = 0;

    // Moving average for smooth FPS display
    double smooth_fps = 30.0;

    while (cap.read(frame)) {
        if (frame.empty()) break;

        // Run object detection
        auto detections = detector->detect(frame);
        const auto& metrics = detector->get_last_metrics();

        if (metrics.total_time_ms > 0.0) {
            smooth_fps = smooth_fps * 0.9 + metrics.fps * 0.1;
        }

        // Count objects per class
        std::map<int, int> counts;
        for (const auto& d : detections) {
            counts[d.class_id]++;
        }

        // Draw bounding boxes
        for (const auto& det : detections) {
            cv::Scalar color = class_colors[det.class_id % class_colors.size()];

            cv::Rect box(static_cast<int>(det.bbox.x), 
                         static_cast<int>(det.bbox.y), 
                         static_cast<int>(det.bbox.width), 
                         static_cast<int>(det.bbox.height));

            cv::rectangle(frame, box, color, 2, cv::LINE_AA);

            // Label text
            char label_txt[128];
            snprintf(label_txt, sizeof(label_txt), "%s %.0f%%", 
                     det.class_name.c_str(), det.confidence * 100.0f);

            int base_line = 0;
            cv::Size text_size = cv::getTextSize(label_txt, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base_line);
            int top = std::max(box.y, text_size.height + 6);
            cv::Rect label_box(box.x, top - text_size.height - 6, text_size.width + 8, text_size.height + 6);

            cv::rectangle(frame, label_box, color, cv::FILLED);
            cv::putText(frame, label_txt, cv::Point(box.x + 4, top - 3), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        // Top Status HUD Panel
        cv::Rect hud_panel(0, 0, frame.cols, 55);
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, hud_panel, cv::Scalar(20, 20, 20), cv::FILLED);
        cv::addWeighted(overlay, 0.75, frame, 0.25, 0, frame);

        // FPS & Latency info
        char hud_perf[128];
        snprintf(hud_perf, sizeof(hud_perf), "%.1f FPS | Latency: %.1f ms (CoreML)", 
                 smooth_fps, metrics.total_time_ms);
        cv::putText(frame, hud_perf, cv::Point(15, 24), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // Class counts info
        char hud_counts[256];
        snprintf(hud_counts, sizeof(hud_counts), 
                 "Forklift: %d  |  Person: %d  |  Helmet: %d  |  Vest: %d", 
                 counts[0], counts[2], counts[1], counts[3]);
        cv::putText(frame, hud_counts, cv::Point(15, 47), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // Safety Alert badge (if people detected but missing PPE)
        int persons = counts[2];
        int helmets = counts[1];
        int vests = counts[3];
        if (persons > 0 && (helmets < persons || vests < persons)) {
            cv::putText(frame, "SAFETY ALERT: Missing PPE!", 
                        cv::Point(frame.cols - 330, 36), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 70, 255), 2, cv::LINE_AA);
        } else if (persons > 0) {
            cv::putText(frame, "ALL PPE COMPLIANT", 
                        cv::Point(frame.cols - 270, 36), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(50, 255, 50), 2, cv::LINE_AA);
        }

        cv::imshow(window_name, frame);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 'q' || key == 'Q' || key == 27) { // 27 = ESC
            break;
        } else if (key == 's' || key == 'S') {
            snapshot_count++;
            std::string filename = "snapshot_" + std::to_string(snapshot_count) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "[Snapshot Saved] " << filename << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "\nCamera closed successfully.\n";
    return 0;
}
