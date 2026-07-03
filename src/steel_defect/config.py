from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
YOLO_DIR = DATA_DIR / "yolo"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_ALIASES = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches",
    "crazing": "crazing",
    "inclusion": "inclusion",
    "patches": "patches",
    "pitted_surface": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "scratches": "scratches",
}


def ensure_output_dirs() -> None:
    for path in [FIGURES_DIR, REPORTS_DIR, WEIGHTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

