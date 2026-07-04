#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Existing YOLOv8 runs can stay in runs/detect. These runs add the
# detection-focused ablation used in the report:
# YOLOv8n baseline -> YOLO11n -> YOLO11n at higher resolution ->
# YOLO11n + EMA attention + small-defect loss gains.
"$PYTHON_BIN" scripts/train_yolo.py \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 224 \
  --batch 32 \
  --device 0 \
  --name yolo11n_neu \
  --loss-preset default

"$PYTHON_BIN" scripts/train_yolo.py \
  --model yolo11s.pt \
  --epochs 100 \
  --imgsz 224 \
  --batch 32 \
  --device 0 \
  --name yolo11s_neu \
  --loss-preset default

"$PYTHON_BIN" scripts/train_yolo.py \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 320 \
  --batch 32 \
  --device 0 \
  --name yolo11n_img320 \
  --loss-preset default

"$PYTHON_BIN" scripts/train_yolo.py \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 320 \
  --batch 32 \
  --device 0 \
  --name yolo11n_ema_loss320 \
  --attention ema \
  --loss-preset small_defect

"$PYTHON_BIN" scripts/aggregate_results.py
"$PYTHON_BIN" scripts/plot_yolo_training_curves.py
