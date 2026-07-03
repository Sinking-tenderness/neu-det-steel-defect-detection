from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

from steel_defect.config import PROJECT_ROOT


def count_files(root: Path) -> tuple[int, int]:
    images = sum(1 for p in (root / "IMAGES").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    annotations = sum(1 for _ in (root / "ANNOTATIONS").rglob("*.xml"))
    return images, annotations


def copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["IMAGES", "ANNOTATIONS"]:
        target = dst / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src / name, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NEU-DET from a public Kaggle mirror.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "NEU-DET")
    parser.add_argument("--dataset", default="rdsunday/neu-urface-defect-database")
    args = parser.parse_args()

    cache_path = Path(kagglehub.dataset_download(args.dataset))
    source = cache_path / "NEU-DET"
    if not source.exists():
        raise RuntimeError(f"NEU-DET folder not found in downloaded dataset: {cache_path}")
    copy_tree(source, args.out_dir)
    images, annotations = count_files(args.out_dir)
    print(f"Downloaded NEU-DET to {args.out_dir}")
    print(f"images={images} annotations={annotations}")
    if images != 1800 or annotations != 1800:
        raise RuntimeError("Expected 1800 images and 1800 annotations. Please verify the dataset source.")


if __name__ == "__main__":
    main()

