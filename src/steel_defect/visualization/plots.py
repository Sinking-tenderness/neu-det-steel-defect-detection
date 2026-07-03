from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_class_distribution(counts: dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"class": list(counts.keys()), "count": list(counts.values())})
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=df, x="class", y="count", color="#4C78A8")
    plt.xticks(rotation=25, ha="right")
    plt.title("NEU-DET class distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)

