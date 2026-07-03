from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO


def build_app(weights: Path, imgsz: int, conf: float):
    model = YOLO(str(weights))

    def infer(image: Image.Image):
        if image is None:
            return None, "请先上传图片。"
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "input.png"
            image.save(image_path)
            start = time.perf_counter()
            results = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)
            elapsed = (time.perf_counter() - start) * 1000
            result = results[0]
            plotted = Image.fromarray(result.plot()[..., ::-1])
            lines = [f"推理耗时: {elapsed:.2f} ms"]
            if result.boxes is None or len(result.boxes) == 0:
                lines.append("未检测到缺陷。")
            else:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    score = float(box.conf.item())
                    name = result.names[cls_id]
                    lines.append(f"{name}: {score:.3f}")
            return plotted, "\n".join(lines)

    with gr.Blocks(title="NEU-DET YOLOv8 Defect Detection") as demo:
        gr.Markdown("# NEU-DET 钢材表面缺陷检测")
        with gr.Row():
            image = gr.Image(type="pil", label="上传钢材表面图像")
            output = gr.Image(type="pil", label="检测结果")
        text = gr.Textbox(label="类别与置信度", lines=8)
        button = gr.Button("开始检测", variant="primary")
        button.click(infer, inputs=image, outputs=[output, text])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Gradio YOLOv8 demo.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.weights, args.imgsz, args.conf).launch(
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()

