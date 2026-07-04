from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from steel_defect.config import PROJECT_ROOT, REPORTS_DIR, WEIGHTS_DIR, ensure_output_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NEU-DET detection data.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "yolo" / "neu_det.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt, yolov8s.pt, or a custom yaml/pt path.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None, help="Example: 0 for first GPU, cpu for CPU.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    ensure_output_dirs()
    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    run_name = args.name or Path(args.model).stem

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=run_name,
        device=device,
        workers=args.workers,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.05,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        amp=False,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        target = WEIGHTS_DIR / f"{run_name}_best.pt"
        target.write_bytes(best.read_bytes())
        print(f"Copied best weights to {target}")

    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=device,
        project=str(args.project),
        name=f"{run_name}_test",
    )
    row = {
        "model": run_name,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    out_path = REPORTS_DIR / f"{run_name}_yolo_test_metrics.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(row)
    print(f"YOLO metrics written to {out_path}")


if __name__ == "__main__":
    main()
