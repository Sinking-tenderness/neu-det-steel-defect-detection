from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ultralytics import YOLO

from steel_defect.config import CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, WEIGHTS_DIR, ensure_output_dirs
from scripts.train_classifier import build_model


@dataclass(frozen=True)
class Corruption:
    name: str
    label: str


CORRUPTIONS = [
    Corruption("clean", "Clean"),
    Corruption("brightness_down", "Low brightness"),
    Corruption("brightness_up", "High brightness"),
    Corruption("low_contrast", "Low contrast"),
    Corruption("gaussian_noise", "Gaussian noise"),
    Corruption("gaussian_blur", "Gaussian blur"),
]


def corrupt_image(image: np.ndarray, corruption: str) -> np.ndarray:
    if corruption == "clean":
        return image
    if corruption == "brightness_down":
        return cv2.convertScaleAbs(image, alpha=0.75, beta=-25)
    if corruption == "brightness_up":
        return cv2.convertScaleAbs(image, alpha=1.15, beta=30)
    if corruption == "low_contrast":
        mean = image.mean()
        return np.clip((image.astype(np.float32) - mean) * 0.55 + mean, 0, 255).astype(np.uint8)
    if corruption == "gaussian_noise":
        rng = np.random.default_rng(2026)
        noise = rng.normal(0, 18, size=image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if corruption == "gaussian_blur":
        return cv2.GaussianBlur(image, (5, 5), 1.2)
    raise ValueError(f"Unknown corruption: {corruption}")


class CorruptedImageFolder(datasets.ImageFolder):
    def __init__(self, root: Path, corruption: str, image_size: int):
        self.corruption = corruption
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        super().__init__(root, transform=transform)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        image = corrupt_image(image, self.corruption)
        return self.transform(image), target


@torch.inference_mode()
def evaluate_classifier(
    model_name: str,
    checkpoint: Path,
    data_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> list[dict[str, float | str]]:
    state = torch.load(checkpoint, map_location=device)
    model = build_model(model_name, len(CLASS_NAMES), pretrained=False).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    rows: list[dict[str, float | str]] = []
    for corruption in CORRUPTIONS:
        dataset = CorruptedImageFolder(data_dir / "test", corruption.name, image_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        y_true, y_pred = [], []
        start = time.perf_counter()
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())
        seconds_per_image = (time.perf_counter() - start) / max(len(dataset), 1)
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        rows.append(
            {
                "model": model_name,
                "corruption": corruption.name,
                "corruption_label": corruption.label,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "seconds_per_image": seconds_per_image,
            }
        )
    return rows


def make_corrupted_yolo_dataset(source_yolo_dir: Path, out_root: Path, corruption: Corruption) -> Path:
    images_src = source_yolo_dir / "images" / "test"
    labels_src = source_yolo_dir / "labels" / "test"
    dataset_root = out_root / corruption.name
    images_dst = dataset_root / "images" / "test"
    labels_dst = dataset_root / "labels" / "test"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(images_src.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        corrupted = corrupt_image(image, corruption.name)
        cv2.imwrite(str(images_dst / image_path.name), corrupted)
        label_path = labels_src / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, labels_dst / label_path.name)

    yaml_path = dataset_root / "neu_det_robust.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        f"path: {dataset_root.as_posix()}\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def evaluate_yolo(
    model_name: str,
    weights: Path,
    yolo_dir: Path,
    work_dir: Path,
    image_size: int,
    device: str,
) -> list[dict[str, float | str]]:
    model = YOLO(str(weights))
    rows: list[dict[str, float | str]] = []
    for corruption in CORRUPTIONS:
        yaml_path = make_corrupted_yolo_dataset(yolo_dir, work_dir, corruption)
        metrics = model.val(data=str(yaml_path), split="test", imgsz=image_size, device=device, verbose=False)
        rows.append(
            {
                "model": model_name,
                "corruption": corruption.name,
                "corruption_label": corruption.label,
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
        )
    return rows


def plot_robustness(df: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 4.8))
    labels = [c.label for c in CORRUPTIONS]
    x = np.arange(len(labels))
    models = list(df["model"].drop_duplicates())
    width = min(0.8 / max(len(models), 1), 0.25)
    for idx, model in enumerate(models):
        values = []
        sub = df[df["model"] == model].set_index("corruption")
        for corruption in CORRUPTIONS:
            values.append(float(sub.loc[corruption.name, metric]) if corruption.name in sub.index else np.nan)
        offset = (idx - (len(models) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=model)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def resolve_devices(device_arg: str | None) -> tuple[torch.device, str | int]:
    if device_arg is None:
        if torch.cuda.is_available():
            return torch.device("cuda:0"), 0
        return torch.device("cpu"), "cpu"
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}"), int(device_arg)
    return torch.device(device_arg), device_arg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifier and YOLO robustness under synthetic corruptions.")
    parser.add_argument("--classification-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "classification")
    parser.add_argument("--yolo-dir", type=Path, default=PROJECT_ROOT / "data" / "yolo")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-classifiers", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    device, yolo_device = resolve_devices(args.device)

    if not args.skip_classifiers:
        classifier_rows = []
        for model_name in ["resnet50", "mobilenet_v2", "mobilenet_v2_se"]:
            checkpoint = WEIGHTS_DIR / f"{model_name}_best.pt"
            if checkpoint.exists():
                classifier_rows.extend(
                    evaluate_classifier(
                        model_name,
                        checkpoint,
                        args.classification_dir,
                        args.image_size,
                        args.batch_size,
                        args.workers,
                        device,
                    )
                )
        if classifier_rows:
            df = pd.DataFrame(classifier_rows)
            df.to_csv(REPORTS_DIR / "robustness_classification.csv", index=False)
            plot_robustness(df, "accuracy", "Classification robustness under synthetic corruptions", FIGURES_DIR / "robustness_classification_accuracy.png")
            plot_robustness(df, "macro_f1", "Classification macro-F1 robustness", FIGURES_DIR / "robustness_classification_macro_f1.png")
            print(df)

    if not args.skip_yolo:
        yolo_rows = []
        yolo_weights = {
            "yolov8n_neu": PROJECT_ROOT / "runs" / "detect" / "yolov8n_neu-2" / "weights" / "best.pt",
            "yolov8s_neu": PROJECT_ROOT / "runs" / "detect" / "yolov8s_neu" / "weights" / "best.pt",
        }
        work_dir = PROJECT_ROOT / "data" / "robustness_yolo"
        for model_name, weights in yolo_weights.items():
            if weights.exists():
                yolo_rows.extend(evaluate_yolo(model_name, weights, args.yolo_dir, work_dir, args.image_size, str(yolo_device)))
        if yolo_rows:
            df = pd.DataFrame(yolo_rows)
            df.to_csv(REPORTS_DIR / "robustness_detection.csv", index=False)
            plot_robustness(df, "map50", "YOLO mAP50 robustness under synthetic corruptions", FIGURES_DIR / "robustness_detection_map50.png")
            plot_robustness(df, "map50_95", "YOLO mAP50-95 robustness", FIGURES_DIR / "robustness_detection_map50_95.png")
            print(df)


if __name__ == "__main__":
    main()
