# NEU-DET Steel Surface Defect Detection

这是一个用于结课项目的完整工程骨架，目标是基于 NEU-DET 数据集实现钢材表面缺陷的传统机器视觉分类、深度学习分类和 YOLOv8 目标检测。

## 1. 环境安装

建议 Python 3.10 或 3.11。

```powershell
cd D:\gc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

如果你使用 GPU，请优先按 PyTorch 官网命令安装与你 CUDA 版本匹配的 `torch` 和 `torchvision`，再安装其他依赖。

## 2. 数据放置

把 NEU-DET 原始数据放到：

```text
D:\gc\data\raw
```

推荐结构之一：

```text
data/raw/
  Cr/
  In/
  Pa/
  PS/
  RS/
  Sc/
  Annotations/
```

也支持完整类名目录：

```text
crazing/
inclusion/
patches/
pitted_surface/
rolled-in_scale/
scratches/
```

也可以直接用脚本下载 Kaggle 公开镜像：

```powershell
python scripts\download_neu_det.py
```

脚本会写入：

```text
data/raw/NEU-DET/IMAGES
data/raw/NEU-DET/ANNOTATIONS
```

并校验 `1800` 张图像和 `1800` 个 XML 标注。论文和报告中建议引用 NEU 官方页面作为数据集来源，Kaggle 只作为下载镜像。

## 3. 数据准备

```powershell
python scripts\prepare_dataset.py --raw-dir data\raw
```

输出：

```text
data/processed/classification
data/yolo
data/yolo/neu_det.yaml
```

如果 `data/raw` 中没有 VOC XML 标注，脚本会跳过 YOLO 数据集转换，只生成分类数据。

## 4. 数据分析和预处理图

```powershell
python scripts\visualize_dataset.py
```

输出图表在：

```text
outputs/figures
```

包括类别分布图、预处理对比图、各类灰度均值热力图。

## 5. 传统机器视觉分类

```powershell
python scripts\train_traditional.py
```

输出：

```text
outputs/reports/traditional_metrics.csv
outputs/weights/traditional_svm.joblib
outputs/figures/traditional_*_confusion_matrix.png
```

## 6. 深度学习分类

CPU 可以先小 epoch 验证流程：

```powershell
python scripts\train_classifier.py --model mobilenet_v2 --epochs 2 --batch-size 8
```

正式训练建议 GPU：

```powershell
python scripts\train_classifier.py --model resnet50 --epochs 40 --batch-size 32
python scripts\train_classifier.py --model mobilenet_v2 --epochs 40 --batch-size 32
python scripts\train_classifier.py --model mobilenet_v2_se --epochs 40 --batch-size 32
```

输出：

```text
outputs/weights/*_best.pt
outputs/reports/*_metrics.csv
outputs/figures/*_confusion_matrix.png
```

## 7. YOLOv8 目标检测

CPU 小测试：

```powershell
python scripts\train_yolo.py --model yolov8n.pt --epochs 2 --batch 4 --device cpu
```

正式训练建议 GPU：

```powershell
python scripts\train_yolo.py --model yolov8n.pt --epochs 80 --batch 16 --device 0 --name yolov8n_neu
python scripts\train_yolo.py --model yolov8s.pt --epochs 80 --batch 16 --device 0 --name yolov8s_neu
```

输出：

```text
runs/detect/
outputs/weights/yolov8n_neu_best.pt
outputs/reports/yolov8n_neu_yolo_test_metrics.csv
```

## 8. 推理和演示

单张或文件夹预测：

```powershell
python scripts\predict_yolo.py --weights outputs\weights\yolov8n_neu_best.pt --source data\processed\classification\test
```

Gradio 演示：

```powershell
python app\gradio_yolo_demo.py --weights outputs\weights\yolov8n_neu_best.pt
```

浏览器打开：

```text
http://127.0.0.1:7860
```

## 9. ONNX 导出

```powershell
python scripts\export_yolo_onnx.py --weights outputs\weights\yolov8n_neu_best.pt
python scripts\export_classifier_onnx.py --weights outputs\weights\mobilenet_v2_best.pt
```

## 10. 结果汇总

```powershell
python scripts\aggregate_results.py
```

输出：

```text
outputs/reports/classification_comparison.csv
outputs/reports/detection_comparison.csv
outputs/figures/classification_comparison.png
outputs/figures/detection_comparison.png
```

## GPU 建议

传统机器学习不需要 GPU。ResNet50、MobileNetV2、YOLOv8 正式训练建议开 GPU。

最低可用：8GB 显存，`batch=8` 或更低。

推荐开卡：RTX 3090 24GB、RTX 4090 24GB、A10 24GB 这一档。NEU-DET 图像只有 200x200，训练 YOLOv8n/s、ResNet50、MobileNetV2 都很宽裕。

省钱可用：RTX 3060 12GB、T4 16GB。把 YOLO batch 调到 8-16 即可。

环境建议：Ubuntu + Python 3.10 + PyTorch 2.x + CUDA 11.8 或 12.1。AutoDL 里直接选择 PyTorch 镜像即可，不需要 TensorFlow。

如果只是先检查代码流程，可以全程 CPU 跑 1-2 个 epoch。

## AutoDL 训练建议

```bash
git clone <your-repo-url>
cd steel-defect-detection
bash scripts/autodl_prepare.sh
```

本项目当前 GitHub 仓库：

```text
https://github.com/Sinking-tenderness/neu-det-steel-defect-detection
```

数据目录和训练产物默认不进 Git；在 AutoDL 上按上面的命令重新下载和生成即可。

开卡后正式训练：

```bash
bash scripts/autodl_train.sh
```
