from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from steel_defect.config import CLASS_ALIASES, CLASS_NAMES, PROJECT_ROOT
from steel_defect.data.io import copy_file, iter_images


def class_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part in CLASS_ALIASES:
            return CLASS_ALIASES[part]
    stem = path.stem
    for alias, canonical in sorted(CLASS_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if stem == alias or stem.startswith(f"{alias}_"):
            return canonical
    return None


def split_items(items: list[Path], train: float, val: float, seed: int):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val :],
    }


def prepare_classification(raw_dir: Path, out_dir: Path, train: float, val: float, seed: int) -> None:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for image_path in iter_images(raw_dir):
        cls = class_from_path(image_path)
        if cls is not None:
            by_class[cls].append(image_path)

    if not by_class:
        raise RuntimeError(
            f"No class images found under {raw_dir}. Put NEU-DET images in class folders such as Cr/In/Pa."
        )

    for cls, images in sorted(by_class.items()):
        splits = split_items(images, train, val, seed)
        for split, paths in splits.items():
            for src in paths:
                dst = out_dir / "classification" / split / cls / src.name
                copy_file(src, dst)

    counts = {cls: len(paths) for cls, paths in sorted(by_class.items())}
    (out_dir / "classification").mkdir(parents=True, exist_ok=True)
    with (out_dir / "classification" / "counts.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(counts, f, allow_unicode=True, sort_keys=False)
    print(f"Classification dataset written to {out_dir / 'classification'}")
    print(counts)


def find_annotation(image_path: Path, raw_dir: Path, annotation_index: dict[str, Path] | None = None) -> Path | None:
    if annotation_index is not None and image_path.stem in annotation_index:
        return annotation_index[image_path.stem]
    candidates = [
        image_path.with_suffix(".xml"),
        raw_dir / "ANNOTATIONS" / f"{image_path.stem}.xml",
        raw_dir / "Annotations" / f"{image_path.stem}.xml",
        raw_dir / "annotations" / f"{image_path.stem}.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(raw_dir.rglob(f"{image_path.stem}.xml"))
    return matches[0] if matches else None


def voc_xml_to_yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {xml_path}")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        cls = CLASS_ALIASES.get(name, name)
        if cls not in CLASS_NAMES:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        cls_id = CLASS_NAMES.index(cls)
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return lines


def prepare_detection(raw_dir: Path, out_dir: Path, train: float, val: float, seed: int) -> None:
    annotation_index = {path.stem: path for path in raw_dir.rglob("*.xml")}
    image_label_pairs = []
    for image_path in iter_images(raw_dir):
        xml_path = find_annotation(image_path, raw_dir, annotation_index)
        if xml_path is not None:
            image_label_pairs.append((image_path, xml_path))

    if not image_label_pairs:
        print("No VOC XML annotations found; skipping YOLO detection dataset creation.")
        return

    by_class: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for image_path, xml_path in image_label_pairs:
        cls = class_from_path(image_path)
        if cls is None:
            lines = voc_xml_to_yolo(xml_path)
            cls = CLASS_NAMES[int(lines[0].split()[0])] if lines else "unknown"
        by_class[cls].append((image_path, xml_path))

    split_pairs: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    for pairs in by_class.values():
        paths = list(range(len(pairs)))
        rng = random.Random(seed)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * train)
        n_val = int(n * val)
        split_pairs["train"].extend(pairs[i] for i in paths[:n_train])
        split_pairs["val"].extend(pairs[i] for i in paths[n_train : n_train + n_val])
        split_pairs["test"].extend(pairs[i] for i in paths[n_train + n_val :])

    yolo_dir = out_dir / "yolo"
    for split, pairs in split_pairs.items():
        for image_path, xml_path in tqdm(pairs, desc=f"YOLO {split}"):
            dst_image = yolo_dir / "images" / split / image_path.name
            dst_label = yolo_dir / "labels" / split / f"{image_path.stem}.txt"
            copy_file(image_path, dst_image)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(voc_xml_to_yolo(xml_path)) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(yolo_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with (yolo_dir / "neu_det.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    print(f"YOLO dataset written to {yolo_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NEU-DET classification and YOLO datasets.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train + args.val >= 1:
        raise ValueError("--train + --val must be less than 1.0")
    prepare_classification(args.raw_dir, args.out_dir, args.train, args.val, args.seed)
    prepare_detection(args.raw_dir, PROJECT_ROOT / "data", args.train, args.val, args.seed)


if __name__ == "__main__":
    main()
