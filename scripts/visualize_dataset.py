from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from steel_defect.config import CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT, ensure_output_dirs
from steel_defect.data.io import iter_images, read_gray
from steel_defect.data.preprocess import clahe_enhance, contour_boxes, threshold_defect
from steel_defect.visualization.plots import save_class_distribution


def save_preprocess_examples(data_dir: Path, out_path: Path, max_per_class: int = 1) -> None:
    rows = []
    for cls in CLASS_NAMES:
        class_dir = data_dir / "train" / cls
        images = list(iter_images(class_dir))[:max_per_class]
        for image_path in images:
            gray = read_gray(image_path)
            enhanced = clahe_enhance(gray)
            binary = threshold_defect(gray)
            boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for x, y, w, h, _ in contour_boxes(gray):
                cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 0, 255), 1)
            rows.append((cls, gray, enhanced, binary, cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)))

    if not rows:
        raise RuntimeError(f"No images found in {data_dir}")

    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.6 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, 0)
    titles = ["original", "CLAHE", "threshold", "contours"]
    for row_idx, (cls, *images) in enumerate(rows):
        for col_idx, image in enumerate(images):
            axes[row_idx, col_idx].imshow(image, cmap="gray" if col_idx < 3 else None)
            axes[row_idx, col_idx].set_title(f"{cls} - {titles[col_idx]}")
            axes[row_idx, col_idx].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def save_heatmaps(data_dir: Path, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.ravel()
    for idx, cls in enumerate(CLASS_NAMES):
        images = list(iter_images(data_dir / "train" / cls))
        acc = None
        for image_path in tqdm(images, desc=f"heatmap {cls}"):
            gray = read_gray(image_path).astype(np.float32) / 255.0
            acc = gray if acc is None else acc + gray
        if acc is None:
            axes[idx].axis("off")
            continue
        mean = acc / max(len(images), 1)
        axes[idx].imshow(mean, cmap="magma")
        axes[idx].set_title(cls)
        axes[idx].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dataset and preprocessing figures.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "classification",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    counts = Counter()
    for cls in CLASS_NAMES:
        counts[cls] = len(list(iter_images(args.data_dir / "train" / cls)))
    save_class_distribution(dict(counts), FIGURES_DIR / "class_distribution.png")
    save_preprocess_examples(args.data_dir, FIGURES_DIR / "preprocess_examples.png")
    save_heatmaps(args.data_dir, FIGURES_DIR / "class_gray_heatmaps.png")
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

