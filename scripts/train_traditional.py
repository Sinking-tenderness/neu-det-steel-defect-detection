from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from steel_defect.config import CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, WEIGHTS_DIR, ensure_output_dirs
from steel_defect.data.io import iter_images, read_gray
from steel_defect.features.handcrafted import combined_features
from steel_defect.visualization.plots import save_confusion_matrix


def load_split(data_dir: Path, split: str, use_sift: bool):
    features, labels, paths = [], [], []
    for label, cls in enumerate(CLASS_NAMES):
        for image_path in tqdm(list(iter_images(data_dir / split / cls)), desc=f"{split} {cls}"):
            gray = read_gray(image_path)
            features.append(combined_features(gray, use_sift=use_sift))
            labels.append(label)
            paths.append(str(image_path))
    if not features:
        raise RuntimeError(f"No images found in {data_dir / split}")
    return np.vstack(features), np.array(labels), paths


def build_models(pca_components: int):
    return {
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=42)),
                ("clf", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=42)),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=pca_components, random_state=42)),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


def evaluate_model(name: str, model, x, y, split: str) -> dict:
    pred = model.predict(x)
    report = classification_report(
        y,
        pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    save_confusion_matrix(
        y,
        pred,
        CLASS_NAMES,
        FIGURES_DIR / f"traditional_{name}_{split}_confusion_matrix.png",
        f"{name} {split} confusion matrix",
    )
    return {
        "model": name,
        "split": split,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train traditional ML baselines on NEU-DET.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "classification",
    )
    parser.add_argument("--pca-components", type=int, default=120)
    parser.add_argument("--no-sift", action="store_true", help="Use only HOG+GLCM features.")
    args = parser.parse_args()

    ensure_output_dirs()
    use_sift = not args.no_sift
    x_train, y_train, _ = load_split(args.data_dir, "train", use_sift)
    x_val, y_val, _ = load_split(args.data_dir, "val", use_sift)
    x_test, y_test, _ = load_split(args.data_dir, "test", use_sift)

    max_components = min(args.pca_components, x_train.shape[0] - 1, x_train.shape[1])
    rows = []
    for name, model in build_models(max_components).items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        rows.append(evaluate_model(name, model, x_val, y_val, "val"))
        rows.append(evaluate_model(name, model, x_test, y_test, "test"))
        joblib.dump(model, WEIGHTS_DIR / f"traditional_{name}.joblib")

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "traditional_metrics.csv", index=False)
    print(df)
    print(f"Traditional model artifacts written to {WEIGHTS_DIR} and {REPORTS_DIR}")


if __name__ == "__main__":
    main()

