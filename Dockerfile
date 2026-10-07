# ==============================================================================
# Dockerfile for YOLOv12n Industrial AI Safety Monitoring Web Application
# Engine: C++ / ONNX Runtime / OpenCV / FastAPI (Embedded Model)
# Supports: x86_64 & aarch64 (Multi-platform)
# ==============================================================================

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# 1. Install System Dependencies (C++ toolchain, OpenCV, FFmpeg, Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    libopencv-dev \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Official ONNX Runtime C++ Shared Library (v1.18.0)
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-aarch64-1.18.0.tgz"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    echo "Downloading ONNX Runtime for $ARCH from $ORT_URL..." && \
    mkdir -p /opt/onnxruntime && \
    wget -q "$ORT_URL" -O /tmp/ort.tgz && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && \
    cp -r /opt/onnxruntime/include/* /usr/local/include/ && \
    cp -r /opt/onnxruntime/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/ort.tgz

ENV ONNXRUNTIME_ROOT_DIR=/opt/onnxruntime \
    LD_LIBRARY_PATH=/opt/onnxruntime/lib:/usr/local/lib:${LD_LIBRARY_PATH}

# 3. Set Working Directory
WORKDIR /app

# 4. Install Python Dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# 5. Copy Source Code, Embedded Model, Configurations, and Web Assets
COPY CMakeLists.txt /app/
COPY include/ /app/include/
COPY src/ /app/src/
COPY config/ /app/config/
COPY web/ /app/web/
COPY labels.txt /app/
COPY yolo_cpp.py /app/
COPY server.py /app/
COPY run_web.sh /app/

# Embedded YOLOv12n ONNX Model Weights
COPY best.onnx /app/best.onnx
COPY best.onnx.data /app/best.onnx.data

# 6. Compile C++ Shared Library (libyolo_detector.so) inside Linux Container
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime && \
    cmake --build build -j$(nproc) && \
    strip --strip-unneeded build/libyolo_detector.so && \
    chmod +x /app/run_web.sh

# 7. Create Runtime Directories
RUN mkdir -p /app/uploads /app/results

# 8. Expose Port
EXPOSE 8080

# 9. Health Check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# 10. Start Server
CMD ["python3", "server.py", "8080"]
