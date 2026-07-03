#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
pip install -r requirements.txt
pip install -e .

python scripts/download_neu_det.py
python scripts/prepare_dataset.py --raw-dir data/raw

# Default quick production run. Edit epochs/batch/model if the GPU is larger or smaller.
python scripts/train_yolo.py --model yolov8n.pt --epochs 80 --batch 16 --device 0 --name yolov8n_neu

