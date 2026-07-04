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


def filter_detection_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df:
        return df
    df = df.copy()
    for column, default in {
        "imgsz": pd.NA,
        "attention": "none",
        "attention_layers": "",
        "loss_preset": "default",
        "box_gain": 7.5,
        "cls_gain": 0.5,
        "dfl_gain": 1.5,
    }.items():
        if column not in df:
            df[column] = default
    model = df["model"].astype(str)
    keep = model.str.contains("yolo", case=False, na=False)
    keep &= ~model.str.contains("cpu_smoke", case=False, na=False)
    df = df[keep]
    legacy_224 = df["model"].isin(["yolov8n_neu", "yolov8s_neu"])
    df.loc[legacy_224 & df["imgsz"].isna(), "imgsz"] = 224
    df["attention"] = df["attention"].fillna("none")
    df["loss_preset"] = df["loss_preset"].fillna("default")
    df["box_gain"] = df["box_gain"].fillna(7.5)
    df["cls_gain"] = df["cls_gain"].fillna(0.5)
    df["dfl_gain"] = df["dfl_gain"].fillna(1.5)
    preferred_order = [
        "yolov8n_neu",
        "yolov8s_neu",
        "yolo11n_neu",
        "yolo11s_neu",
        "yolo11n_img256",
        "yolo11n_img320",
        "yolo11n_ema224",
        "yolo11n_loss224",
        "yolo11n_ema_loss224",
        "yolo11n_ema_loss320",
    ]
    order = {name: idx for idx, name in enumerate(preferred_order)}
    df["_order"] = df["model"].map(order).fillna(len(order))
    return df.sort_values(["_order", "model"]).drop(columns="_order").reset_index(drop=True)


def filter_classification_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "model" not in df:
        return df
    df = df.copy()
    model = df["model"].astype(str)
    keep = model.isin(["svm", "random_forest", "knn", "resnet50"])
    columns = ["model", "split", "accuracy", "macro_f1", "weighted_f1", "seconds_per_image"]
    available_columns = [col for col in columns if col in df.columns]
    return df[keep][available_columns].reset_index(drop=True)


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
        classifiers = classifiers[classifiers.get("model", pd.Series(dtype=str)) == "resnet50"]
    classification = (
        filter_classification_runs(pd.concat([traditional, classifiers], ignore_index=True))
        if not traditional.empty or not classifiers.empty
        else pd.DataFrame()
    )
    yolo = filter_detection_runs(read_csvs("*_yolo_test_metrics.csv"))

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
