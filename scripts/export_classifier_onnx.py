from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.train_classifier import build_model
from steel_defect.config import CLASS_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a classification checkpoint to ONNX.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location="cpu")
    model_name = checkpoint["model"]
    model = build_model(model_name, len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)
    output = args.output or args.weights.with_suffix(".onnx")
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported classifier ONNX model to {output}")


if __name__ == "__main__":
    main()

