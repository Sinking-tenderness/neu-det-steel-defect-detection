from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from steel_defect.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, ensure_output_dirs


def read_csvs(pattern: str) -> pd.DataFrame:
    frames = []
    for path in REPORTS_DIR.glob(pattern):
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_classification(df: pd.DataFrame, out_path: Path) -> None:
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        return
    plt.figure(figsize=(8, 4.5))
    melted = test_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.title("Classification model comparison")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_detection(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 4.5))
    melted = df.melt(
        id_vars=["model"],
        value_vars=["precision", "recall", "map50", "map50_95"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=melted, x="model", y="score", hue="metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.title("YOLO detection model comparison")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics and draw comparison charts.")
    parser.parse_args()
    ensure_output_dirs()

    traditional = read_csvs("traditional_metrics.csv")
    classifiers = read_csvs("*_metrics.csv")
    if not classifiers.empty:
        classifiers = classifiers[
            ~classifiers.get("model", pd.Series(dtype=str)).astype(str).str.contains("yolo", case=False, na=False)
        ]
        classifiers = classifiers[classifiers.get("model", pd.Series(dtype=str)) != "svm"]
        classifiers = classifiers[classifiers.get("model", pd.Series(dtype=str)) != "random_forest"]
        classifiers = classifiers[classifiers.get("model", pd.Series(dtype=str)) != "knn"]
    classification = pd.concat([traditional, classifiers], ignore_index=True) if not traditional.empty or not classifiers.empty else pd.DataFrame()
    yolo = read_csvs("*_yolo_test_metrics.csv")

    if not classification.empty:
        classification.to_csv(REPORTS_DIR / "classification_comparison.csv", index=False)
        plot_classification(classification, FIGURES_DIR / "classification_comparison.png")
    if not yolo.empty:
        yolo.to_csv(REPORTS_DIR / "detection_comparison.csv", index=False)
        plot_detection(yolo, FIGURES_DIR / "detection_comparison.png")
    print(f"Aggregated reports written to {REPORTS_DIR}")
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
