#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Existing YOLOv8n/YOLOv8s runs were trained at imgsz=224 by autodl_train.sh.
# Keep the detection ablation orthogonal for the report:
# 1) architecture comparison at imgsz=224
# 2) resolution comparison for YOLO11n
# 3) EMA attention gain at fixed YOLO11n@224
# 4) small-defect loss gain at fixed YOLO11n@224
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
  --imgsz 256 \
  --batch 32 \
  --device 0 \
  --name yolo11n_img256 \
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
  --imgsz 224 \
  --batch 32 \
  --device 0 \
  --name yolo11n_ema224 \
  --attention ema \
  --loss-preset default

"$PYTHON_BIN" scripts/train_yolo.py \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 224 \
  --batch 32 \
  --device 0 \
  --name yolo11n_loss224 \
  --loss-preset small_defect

"$PYTHON_BIN" scripts/train_yolo.py \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 224 \
  --batch 32 \
  --device 0 \
  --name yolo11n_ema_loss224 \
  --attention ema \
  --loss-preset small_defect

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
