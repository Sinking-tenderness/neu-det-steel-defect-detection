from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from steel_defect.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLOv8 checkpoint to ONNX.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    exported = model.export(format="onnx", imgsz=args.imgsz, dynamic=args.dynamic, simplify=True)
    print(f"Exported ONNX model: {PROJECT_ROOT / exported if not Path(exported).is_absolute() else exported}")


if __name__ == "__main__":
    main()

