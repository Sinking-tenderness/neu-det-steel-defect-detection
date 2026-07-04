#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/train_yolo.py --model yolov8n.pt --epochs 100 --batch 32 --device 0 --name yolov8n_neu
"$PYTHON_BIN" scripts/train_yolo.py --model yolov8s.pt --epochs 100 --batch 32 --device 0 --name yolov8s_neu

"$PYTHON_BIN" scripts/train_classifier.py --model resnet50 --epochs 40 --batch-size 32
"$PYTHON_BIN" scripts/train_classifier.py --model mobilenet_v2 --epochs 40 --batch-size 32
"$PYTHON_BIN" scripts/train_classifier.py --model mobilenet_v2_se --epochs 40 --batch-size 32

"$PYTHON_BIN" scripts/aggregate_results.py

