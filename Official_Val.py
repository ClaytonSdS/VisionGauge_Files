"""
Roda a validação oficial do YOLO (ultralytics .val()) para os 3 modelos e salva
os resultados (métricas, matriz de confusão, curvas P/R, etc.) em:

  dataset/testing/Resultados Detetor/official_val/{tag}/
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

DATA_YAML = str(
    Path(__file__).resolve().parent
    / r"dataset\UTM_TEST_DETECTION.v1-utm_detection_test_set.yolov5pytorch\data.yaml"
)

MODELS = [
    ("yolov8s",  r"models\Yolo_v8\weights\best.pt"),
    ("yolov11s", r"models\Yolov11s\weights\best.pt"),
    ("yolo26s",  r"models\Yolo_v26s\weights\best.pt"),
]

OUTPUT_BASE = r"dataset\testing\Resultados Detetor\official_val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validacao oficial YOLO para os 3 modelos.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf",  type=float, default=0.25)
    parser.add_argument("--iou",   type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = 0 if torch.cuda.is_available() else "cpu"

    for tag, model_rel in MODELS:
        model_path = (root / model_rel).resolve()
        output_dir = (root / OUTPUT_BASE / tag).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Modelo: {tag}  ({model_path})")
        print(f"Saida:  {output_dir}")
        print(f"{'='*60}")

        model = YOLO(str(model_path))
        metrics = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            batch=args.batch,
            device=device,
            save_json=False,
            plots=True,
            project=str(output_dir.parent),
            name=tag,
            exist_ok=True,
            verbose=True,
        )

        # Exibe métricas principais
        print(f"\n--- Metricas {tag} ---")
        print(f"Precision (B): {metrics.box.mp:.4f}")
        print(f"Recall    (B): {metrics.box.mr:.4f}")
        print(f"mAP50     (B): {metrics.box.map50:.4f}")
        print(f"mAP50-95  (B): {metrics.box.map:.4f}")

        # Salva relatório TXT
        txt_path = output_dir / f"{tag}_metrics.txt"
        txt_path.write_text(
            "\n".join([
                f"Model: {model_path}",
                f"Data:  {DATA_YAML}",
                f"imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}",
                "",
                f"Precision (B): {metrics.box.mp:.6f}",
                f"Recall    (B): {metrics.box.mr:.6f}",
                f"mAP50     (B): {metrics.box.map50:.6f}",
                f"mAP50-95  (B): {metrics.box.map:.6f}",
            ]),
            encoding="utf-8",
        )
        print(f"Relatorio salvo em: {txt_path}")

    print("\nConcluido.")


if __name__ == "__main__":
    main()
