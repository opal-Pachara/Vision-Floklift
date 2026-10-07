#!/usr/bin/env python3
"""
Model Export Script for C++ Inference Optimization
Exports YOLOv12 / YOLOv11 PyTorch checkpoint (best.pt) to high-performance ONNX.
"""

import sys
import os
import argparse

# Monkey-patch ultralytics requirements check to avoid unnecessary pip conflicts
try:
    import ultralytics.utils.checks as checks
    checks.check_requirements = lambda *args, **kwargs: True
except ImportError:
    pass

from ultralytics import YOLO

def export_model(weights_path="best.pt", output_format="onnx", imgsz=640, opset=17, half=False):
    if not os.path.exists(weights_path):
        print(f"Error: Model file '{weights_path}' does not exist.")
        sys.exit(1)

    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)
    
    print(f"Task: {model.task}")
    print(f"Class names: {model.names}")
    
    # Save labels.txt
    labels_file = "labels.txt"
    with open(labels_file, "w") as f:
        for idx in sorted(model.names.keys()):
            f.write(f"{model.names[idx]}\n")
    print(f"Saved class labels to {labels_file}")

    print(f"Exporting to {output_format.upper()} (imgsz={imgsz}, opset={opset}, half={half})...")
    export_path = model.export(
        format=output_format,
        imgsz=imgsz,
        opset=opset,
        half=half,
        dynamic=False,
        simplify=False,
        nms=False
    )
    print(f"Export completed successfully: {export_path}")
    return export_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX for C++ inference")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to .pt weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension (default: 640)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--half", action="store_true", help="Export as FP16 (half precision)")
    args = parser.parse_args()

    export_model(args.weights, imgsz=args.imgsz, opset=args.opset, half=args.half)
