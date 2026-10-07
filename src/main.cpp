#include "yolo_detector.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#else
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

void print_usage(const char* prog_name) {
    std::cout << "\n======================================================\n";
    std::cout << "  YOLOv12 C++ High-Performance Inference Engine\n";
    std::cout << "======================================================\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <path>       Path to ONNX model file (default: best.onnx)\n";
    std::cout << "  --labels <path>      Path to labels.txt (default: labels.txt)\n";
    std::cout << "  --image <path>       Run inference on single image file\n";
    std::cout << "  --output <path>      Output path for annotated image/video (default: output.jpg)\n";
#ifdef HAVE_OPENCV
    std::cout << "  --video <path/index> Run inference on video file or webcam (e.g. 0 or video.mp4)\n";
    std::cout << "  --no-display         Do not open GUI display window during video processing\n";
#endif
    std::cout << "  --benchmark [N]      Run benchmark mode for N iterations (default: 100)\n";
    std::cout << "  --conf <float>       Confidence threshold [0.0 - 1.0] (default: 0.25)\n";
    std::cout << "  --nms <float>        NMS IoU threshold [0.0 - 1.0] (default: 0.45)\n";
    std::cout << "  --threads <int>      Intra-op thread count (default: 4)\n";
    std::cout << "  --coreml             Enable Apple CoreML Execution Provider (macOS)\n";
    std::cout << "  --cpu                Force CPU Execution Provider\n";
    std::cout << "  --help               Display this help message\n";
    std::cout << "======================================================\n" << std::endl;
}

int main(int argc, char** argv) {
    yolo::ModelConfig config;
    config.model_path = "best.onnx";
    config.labels_path = "labels.txt";
    config.conf_threshold = 0.25f;
    config.nms_threshold = 0.45f;
    config.num_threads = 4;
    config.use_coreml = true;

    std::string image_path;
    std::string video_path;
    std::string output_path = "output.jpg";
    bool run_benchmark = false;
    int benchmark_runs = 100;
    bool no_display = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--labels" && i + 1 < argc) {
            config.labels_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            config.conf_threshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            config.nms_threshold = std::stof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            run_benchmark = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                benchmark_runs = std::stoi(argv[++i]);
            }
        } else if (arg == "--no-display") {
            no_display = true;
        } else if (arg == "--coreml") {
            config.use_coreml = true;
        } else if (arg == "--cpu") {
            config.use_coreml = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Initialize detector
    std::unique_ptr<yolo::YoloDetector> detector;
    try {
        detector = std::make_unique<yolo::YoloDetector>(config);
    } catch (const std::exception& e) {
        std::cerr << "[Error] Failed to initialize YoloDetector: " << e.what() << std::endl;
        return 1;
    }

    // 1. Benchmark Mode
    if (run_benchmark) {
        std::cout << "\n======================================================\n";
        std::cout << "           RUNNING LATENCY BENCHMARK                  \n";
        std::cout << "======================================================\n";
        std::cout << "Iterations: " << benchmark_runs << " runs\n";
        std::cout << "Model: " << config.model_path << "\n";
        std::cout << "Threads: " << config.num_threads << "\n";

        detector->warmup(5);

        // Prepare dummy input
        std::vector<uint8_t> dummy_raw(config.input_width * config.input_height * 3, 128);
        std::vector<double> pre_times, inf_times, post_times, total_times;
        pre_times.reserve(benchmark_runs);
        inf_times.reserve(benchmark_runs);
        post_times.reserve(benchmark_runs);
        total_times.reserve(benchmark_runs);

        for (int i = 0; i < benchmark_runs; ++i) {
            detector->detect_raw(dummy_raw.data(), config.input_width, config.input_height, 3, true);
            const auto& m = detector->get_last_metrics();
            pre_times.push_back(m.preprocess_time_ms);
            inf_times.push_back(m.inference_time_ms);
            post_times.push_back(m.postprocess_time_ms);
            total_times.push_back(m.total_time_ms);
        }

        auto avg = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        auto percentile = [](std::vector<double> v, double p) {
            std::sort(v.begin(), v.end());
            size_t idx = static_cast<size_t>(p * v.size());
            return v[std::min(idx, v.size() - 1)];
        };

        double avg_pre = avg(pre_times);
        double avg_inf = avg(inf_times);
        double avg_post = avg(post_times);
        double avg_total = avg(total_times);
        double fps = 1000.0 / avg_total;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n------------------------------------------------------\n";
        std::cout << "  Preprocess Latency:  " << std::setw(6) << avg_pre << " ms\n";
        std::cout << "  Inference Latency:   " << std::setw(6) << avg_inf << " ms\n";
        std::cout << "  Postprocess Latency: " << std::setw(6) << avg_post << " ms\n";
        std::cout << "  Total Latency (Avg): " << std::setw(6) << avg_total << " ms\n";
        std::cout << "  Total Latency (P50): " << std::setw(6) << percentile(total_times, 0.50) << " ms\n";
        std::cout << "  Total Latency (P90): " << std::setw(6) << percentile(total_times, 0.90) << " ms\n";
        std::cout << "  Total Latency (P99): " << std::setw(6) << percentile(total_times, 0.99) << " ms\n";
        std::cout << "------------------------------------------------------\n";
        std::cout << "  Throughput (FPS):    " << std::setw(6) << fps << " frames/second\n";
        std::cout << "======================================================\n\n";
        return 0;
    }

#ifdef HAVE_OPENCV
    // 2. Single Image Inference with OpenCV
    if (!image_path.empty()) {
        std::cout << "\n[Running Image Inference] " << image_path << std::endl;
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "[Error] Could not load image: " << image_path << std::endl;
            return 1;
        }

        detector->warmup(1);
        auto detections = detector->detect(img);
        const auto& metrics = detector->get_last_metrics();

        std::cout << "\nFound " << detections.size() << " detections:\n";
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            std::cout << "  [" << i + 1 << "] " << d.class_name 
                      << " (" << std::fixed << std::setprecision(1) << (d.confidence * 100.0f) << "%) "
                      << "bbox: [" << d.bbox.x << ", " << d.bbox.y << ", " 
                      << d.bbox.width << ", " << d.bbox.height << "]\n";
        }

        std::cout << "\nLatency Breakdown:\n";
        std::cout << "  Preprocess:  " << metrics.preprocess_time_ms << " ms\n";
        std::cout << "  Inference:   " << metrics.inference_time_ms << " ms\n";
        std::cout << "  Postprocess: " << metrics.postprocess_time_ms << " ms\n";
        std::cout << "  Total:       " << metrics.total_time_ms << " ms (" << metrics.fps << " FPS)\n";

        detector->draw_detections(img, detections, true);
        cv::imwrite(output_path, img);
        std::cout << "\nSaved annotated result to: " << output_path << std::endl;
        return 0;
    }
#else
    // 2. Single Image Inference with stb_image (No OpenCV needed)
    if (!image_path.empty()) {
        std::cout << "\n[Running Image Inference (STB)] " << image_path << std::endl;
        int img_w = 0, img_h = 0, channels = 0;
        uint8_t* data = stbi_load(image_path.c_str(), &img_w, &img_h, &channels, 3);
        if (!data) {
            std::cerr << "[Error] Failed to load image: " << image_path << std::endl;
            return 1;
        }

        detector->warmup(1);
        auto detections = detector->detect_raw(data, img_w, img_h, 3, false); // RGB format
        const auto& metrics = detector->get_last_metrics();

        std::cout << "\nFound " << detections.size() << " detections:\n";
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& d = detections[i];
            std::cout << "  [" << i + 1 << "] " << d.class_name 
                      << " (" << std::fixed << std::setprecision(1) << (d.confidence * 100.0f) << "%) "
                      << "bbox: [" << d.bbox.x << ", " << d.bbox.y << ", " 
                      << d.bbox.width << ", " << d.bbox.height << "]\n";
        }

        std::cout << "\nLatency Breakdown:\n";
        std::cout << "  Preprocess:  " << metrics.preprocess_time_ms << " ms\n";
        std::cout << "  Inference:   " << metrics.inference_time_ms << " ms\n";
        std::cout << "  Postprocess: " << metrics.postprocess_time_ms << " ms\n";
        std::cout << "  Total:       " << metrics.total_time_ms << " ms (" << metrics.fps << " FPS)\n";

        // Draw bounding boxes on raw RGB buffer
        for (const auto& det : detections) {
            int x1 = std::max(0, static_cast<int>(det.bbox.x));
            int y1 = std::max(0, static_cast<int>(det.bbox.y));
            int x2 = std::min(img_w - 1, static_cast<int>(det.bbox.right()));
            int y2 = std::min(img_h - 1, static_cast<int>(det.bbox.bottom()));

            uint8_t color_r = (det.class_id == 0) ? 255 : (det.class_id == 1) ? 255 : 0;
            uint8_t color_g = (det.class_id == 0) ? 140 : (det.class_id == 1) ? 255 : 255;
            uint8_t color_b = (det.class_id == 2) ? 255 : 0;

            for (int t = 0; t < 3; ++t) {
                int ty1 = std::min(img_h - 1, y1 + t);
                int ty2 = std::max(0, y2 - t);
                for (int x = x1; x <= x2; ++x) {
                    int idx1 = (ty1 * img_w + x) * 3;
                    int idx2 = (ty2 * img_w + x) * 3;
                    data[idx1] = color_r; data[idx1 + 1] = color_g; data[idx1 + 2] = color_b;
                    data[idx2] = color_r; data[idx2 + 1] = color_g; data[idx2 + 2] = color_b;
                }
                int tx1 = std::min(img_w - 1, x1 + t);
                int tx2 = std::max(0, x2 - t);
                for (int y = y1; y <= y2; ++y) {
                    int idx1 = (y * img_w + tx1) * 3;
                    int idx2 = (y * img_w + tx2) * 3;
                    data[idx1] = color_r; data[idx1 + 1] = color_g; data[idx1 + 2] = color_b;
                    data[idx2] = color_r; data[idx2 + 1] = color_g; data[idx2 + 2] = color_b;
                }
            }
        }

        stbi_write_jpg(output_path.c_str(), img_w, img_h, 3, data, 95);
        stbi_image_free(data);
        std::cout << "\nSaved annotated result to: " << output_path << std::endl;
        return 0;
    }
#endif

#ifdef HAVE_OPENCV
    // 3. Video / Webcam Inference
    if (!video_path.empty()) {
        cv::VideoCapture cap;
        bool is_camera = false;
        try {
            int cam_id = std::stoi(video_path);
            cap.open(cam_id);
            is_camera = true;
        } catch (...) {
            cap.open(video_path);
        }

        if (!cap.isOpened()) {
            std::cerr << "[Error] Could not open video/camera source: " << video_path << std::endl;
            return 1;
        }

        int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps_source = cap.get(cv::CAP_PROP_FPS);
        if (fps_source <= 0) fps_source = 30.0;

        std::cout << "\nProcessing video stream: " << frame_w << "x" << frame_h 
                  << " @ " << fps_source << " FPS\n";

        cv::VideoWriter writer;
        if (!is_camera) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(output_path, fourcc, fps_source, cv::Size(frame_w, frame_h));
            std::cout << "Saving output video to: " << output_path << std::endl;
        }

        cv::Mat frame;
        int frame_count = 0;
        double total_time = 0.0;

        while (cap.read(frame)) {
            if (frame.empty()) break;

            auto detections = detector->detect(frame);
            detector->draw_detections(frame, detections, true);

            total_time += detector->get_last_metrics().total_time_ms;
            frame_count++;

            if (writer.isOpened()) {
                writer.write(frame);
            }

            if (!no_display) {
                cv::imshow("YOLOv12 C++ Detector", frame);
                char key = static_cast<char>(cv::waitKey(1));
                if (key == 27 || key == 'q') { // ESC or q
                    break;
                }
            }
        }

        cap.release();
        if (writer.isOpened()) writer.release();
        cv::destroyAllWindows();

        if (frame_count > 0) {
            std::cout << "\nProcessed " << frame_count << " frames in " 
                      << total_time << " ms (Average: " << (1000.0 * frame_count / total_time) 
                      << " FPS)\n";
        }
        return 0;
    }
#else
    if (!video_path.empty()) {
        std::cerr << "[Notice] Video and Webcam features require OpenCV. Please compile with OpenCV installed." << std::endl;
        return 1;
    }
#endif

    // If no action specified, default to running benchmark
    std::cout << "No image or video specified. Running default benchmark (100 runs)...\n";
    std::cout << "Run '" << argv[0] << " --help' for options.\n";

    detector->warmup(5);
    std::vector<uint8_t> dummy_raw(config.input_width * config.input_height * 3, 128);
    for (int i = 0; i < 50; ++i) {
        detector->detect_raw(dummy_raw.data(), config.input_width, config.input_height, 3, true);
    }
    const auto& m = detector->get_last_metrics();
    std::cout << "\nBenchmark Result:\n";
    std::cout << "  Preprocess:  " << m.preprocess_time_ms << " ms\n";
    std::cout << "  Inference:   " << m.inference_time_ms << " ms\n";
    std::cout << "  Postprocess: " << m.postprocess_time_ms << " ms\n";
    std::cout << "  Total Latency: " << m.total_time_ms << " ms (" << m.fps << " FPS)\n";

    return 0;
}
