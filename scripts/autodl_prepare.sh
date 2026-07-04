#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install -U pip
"$PYTHON_BIN" -m pip install -e . --no-deps
"$PYTHON_BIN" -m pip install \
  "numpy>=1.23,<2" \
  "opencv-python==4.10.0.84" \
  "opencv-python-headless==4.10.0.84" \
  "opencv-contrib-python==4.10.0.84" \
  "scikit-image>=0.21" \
  "scikit-learn>=1.3" \
  "pandas>=2.0" \
  "seaborn>=0.12" \
  "joblib>=1.3" \
  "tqdm>=4.66" \
  "PyYAML>=6.0" \
  "kagglehub>=0.3.0" \
  "ultralytics>=8.2,<8.5"

if [ ! -d data/raw/NEU-DET ]; then
  "$PYTHON_BIN" scripts/download_neu_det.py
fi

"$PYTHON_BIN" scripts/prepare_dataset.py --raw-dir data/raw
"$PYTHON_BIN" -m compileall src scripts app

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

checks = {
    "classification_train": Path("data/processed/classification/train"),
    "classification_val": Path("data/processed/classification/val"),
    "classification_test": Path("data/processed/classification/test"),
    "yolo_train_images": Path("data/yolo/images/train"),
    "yolo_val_images": Path("data/yolo/images/val"),
    "yolo_test_images": Path("data/yolo/images/test"),
}
for name, path in checks.items():
    print(name, len([p for p in path.rglob("*") if p.is_file()]))
PY

