cmake_minimum_required(VERSION 3.16)
project(yolo_vision_logistic VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization Flags for Maximum Speed
if(MSVC)
    add_compile_options(/O2 /Oi /Ot /GL /fp:fast)
else()
    add_compile_options(-O3 -fPIC -Wall -Wextra)
    # Architecture-specific SIMD vectorization optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
        message(STATUS "Configuring ARM / Apple Silicon optimizations")
    else()
        if(ENABLE_NATIVE_OPT)
            add_compile_options(-march=native)
        endif()
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 1. Locate ONNX Runtime
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Path to ONNX Runtime installation")
set(ORT_INCLUDE_DIRS "")
set(ORT_LIBRARIES "")

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(ORT_PKG QUIET libonnxruntime)
    if(ORT_PKG_FOUND)
        set(ORT_INCLUDE_DIRS ${ORT_PKG_INCLUDE_DIRS})
        set(ORT_LIBRARIES ${ORT_PKG_LIBRARIES})
        set(ORT_LIBRARY_DIRS ${ORT_PKG_LIBRARY_DIRS})
    endif()
endif()

if(NOT ONNXRUNTIME_ROOT_DIR AND DEFINED ENV{ONNXRUNTIME_ROOT_DIR})
    set(ONNXRUNTIME_ROOT_DIR $ENV{ONNXRUNTIME_ROOT_DIR})
endif()

if(NOT ORT_INCLUDE_DIRS)
    # Direct path check if ONNXRUNTIME_ROOT_DIR or standard paths exist
    foreach(_candidate_root "${ONNXRUNTIME_ROOT_DIR}" "$ENV{ONNXRUNTIME_ROOT_DIR}" "/opt/onnxruntime" "/usr/local" "/opt/homebrew" "/usr")
        if(_candidate_root AND EXISTS "${_candidate_root}/include/onnxruntime_cxx_api.h")
            set(ORT_INCLUDE_DIRS "${_candidate_root}/include")
            if(EXISTS "${_candidate_root}/lib/libonnxruntime.so")
                set(ORT_LIBRARIES "${_candidate_root}/lib/libonnxruntime.so")
            elseif(EXISTS "${_candidate_root}/lib/libonnxruntime.dylib")
                set(ORT_LIBRARIES "${_candidate_root}/lib/libonnxruntime.dylib")
            endif()
            if(ORT_LIBRARIES)
                break()
            endif()
        elseif(_candidate_root AND EXISTS "${_candidate_root}/include/onnxruntime/onnxruntime_cxx_api.h")
            set(ORT_INCLUDE_DIRS "${_candidate_root}/include/onnxruntime")
            if(EXISTS "${_candidate_root}/lib/libonnxruntime.so")
                set(ORT_LIBRARIES "${_candidate_root}/lib/libonnxruntime.so")
            elseif(EXISTS "${_candidate_root}/lib/libonnxruntime.dylib")
                set(ORT_LIBRARIES "${_candidate_root}/lib/libonnxruntime.dylib")
            endif()
            if(ORT_LIBRARIES)
                break()
            endif()
        endif()
    endforeach()
endif()

if(NOT ORT_INCLUDE_DIRS OR NOT ORT_LIBRARIES)
    find_path(ORT_INCLUDE_DIRS
        NAMES onnxruntime_cxx_api.h
        PATHS
            ${ONNXRUNTIME_ROOT_DIR}/include
            $ENV{ONNXRUNTIME_ROOT_DIR}/include
            /opt/onnxruntime/include
            /usr/local/include
            /usr/local/include/onnxruntime
            /opt/homebrew/include/onnxruntime
            /usr/include
            /usr/include/onnxruntime
    )
    find_library(ORT_LIBRARIES
        NAMES onnxruntime
        PATHS
            ${ONNXRUNTIME_ROOT_DIR}/lib
            $ENV{ONNXRUNTIME_ROOT_DIR}/lib
            /opt/onnxruntime/lib
            /usr/local/lib
            /opt/homebrew/lib
            /usr/lib
            /usr/lib/aarch64-linux-gnu
            /usr/lib/x86_64-linux-gnu
    )
endif()

if(NOT ORT_INCLUDE_DIRS OR NOT ORT_LIBRARIES)
    message(FATAL_ERROR "ONNX Runtime not found! Please set ONNXRUNTIME_ROOT_DIR or install via: brew install onnxruntime")
else()
    message(STATUS "Found ONNX Runtime:")
    message(STATUS "  Include: ${ORT_INCLUDE_DIRS}")
    message(STATUS "  Library: ${ORT_LIBRARIES}")
    include_directories(${ORT_INCLUDE_DIRS})
    if(ORT_LIBRARY_DIRS)
        link_directories(${ORT_LIBRARY_DIRS})
    endif()
endif()

# 2. Locate OpenCV (Optional / Recommended)
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio highgui)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}:")
    message(STATUS "  Include: ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_compile_definitions(HAVE_OPENCV)
else()
    message(WARNING "OpenCV not found. Building without OpenCV video/GUI display capabilities.")
endif()

# 3. macOS Frameworks
if(APPLE)
    find_library(COREML_FRAMEWORK CoreML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
endif()

# Source files
set(CORE_SOURCES
    src/yolo_detector.cpp
)

# Build reusable detector library (STATIC)
add_library(yolo_detector STATIC ${CORE_SOURCES})
target_include_directories(yolo_detector PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ORT_INCLUDE_DIRS}
)
target_link_libraries(yolo_detector PUBLIC ${ORT_LIBRARIES})
if(OpenCV_FOUND)
    target_link_libraries(yolo_detector PUBLIC ${OpenCV_LIBS})
endif()
if(APPLE AND COREML_FRAMEWORK)
    target_link_libraries(yolo_detector PUBLIC ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# Build shared library for Python ctypes / C API (produces libyolo_detector.dylib or .so)
add_library(yolo_detector_shared SHARED 
    src/yolo_detector.cpp
    src/yolo_c_api.cpp
)
set_target_properties(yolo_detector_shared PROPERTIES OUTPUT_NAME "yolo_detector")
target_include_directories(yolo_detector_shared PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ORT_INCLUDE_DIRS}
)
target_link_libraries(yolo_detector_shared PUBLIC ${ORT_LIBRARIES})
if(OpenCV_FOUND)
    target_link_libraries(yolo_detector_shared PUBLIC ${OpenCV_LIBS})
endif()
if(APPLE AND COREML_FRAMEWORK)
    target_link_libraries(yolo_detector_shared PUBLIC ${COREML_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# Build standalone CLI executable
add_executable(yolo_cpp src/main.cpp)
target_link_libraries(yolo_cpp PRIVATE yolo_detector)

# Build simple live camera executable (if OpenCV is present)
if(OpenCV_FOUND)
    add_executable(run_camera src/run_camera.cpp)
    target_link_libraries(run_camera PRIVATE yolo_detector)
    message(STATUS "Configured target: run_camera")
endif()

message(STATUS "Configured target: yolo_cpp")
