from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from steel_defect.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, ensure_output_dirs


METRICS = {
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "train/box_loss": "Train box loss",
    "train/cls_loss": "Train cls loss",
    "train/dfl_loss": "Train DFL loss",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def collect_results(runs_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(runs_dir.glob("*/results.csv")):
        df = normalize_columns(pd.read_csv(path))
        if "epoch" not in df:
            continue
        df["model"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_metric(df: pd.DataFrame, metric: str, label: str, out_path: Path) -> None:
    if metric not in df:
        return
    plt.figure(figsize=(8, 4.6))
    for model_name, sub in df.groupby("model", sort=False):
        plt.plot(sub["epoch"], sub[metric], label=model_name, linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.title(f"YOLO training curve: {label}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect YOLO results.csv files and plot per-epoch curves.")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs" / "detect")
    args = parser.parse_args()

    ensure_output_dirs()
    df = collect_results(args.runs_dir)
    if df.empty:
        print(f"No YOLO results.csv files found under {args.runs_dir}")
        return
    out_csv = REPORTS_DIR / "yolo_training_curves.csv"
    df.to_csv(out_csv, index=False)
    for metric, label in METRICS.items():
        plot_metric(df, metric, label, FIGURES_DIR / f"yolo_curve_{metric.replace('/', '_').replace('(', '').replace(')', '')}.png")
    print(f"YOLO training curves written to {out_csv} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
