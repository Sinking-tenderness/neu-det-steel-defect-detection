from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from steel_defect.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference and save visualized predictions.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "outputs" / "predictions")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project),
        name=args.weights.stem,
        save=True,
        save_txt=True,
        save_conf=True,
    )
    print(f"Saved {len(results)} prediction result(s) under {args.project}")


if __name__ == "__main__":
    main()

