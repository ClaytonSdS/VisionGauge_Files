import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera imagens com bounding boxes para um modelo YOLO.")
    parser.add_argument("--model", required=True, help="Caminho do modelo .pt")
    parser.add_argument("--csv", default=r"dataset\testing\dataset_testing_paths.csv", help="CSV com coluna 'file'")
    parser.add_argument("--tag", required=True, help="Prefixo do nome do arquivo, ex: yolov8s")
    parser.add_argument("--output_dir", required=True, help="Diretorio de saida das imagens")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho de inferencia")
    parser.add_argument("--conf", type=float, default=0.25, help="Threshold de confianca")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    model_path = (root_dir / args.model).resolve()
    csv_path = (root_dir / args.csv).resolve()
    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    if "file" not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'file'.")

    paths = []
    for raw in df["file"].astype(str).tolist():
        p = Path(raw)
        if not p.is_absolute():
            p = (root_dir / p).resolve()
        paths.append(str(p))

    total = len(paths)
    if total == 0:
        raise ValueError("Nenhuma imagem encontrada no CSV.")

    print(f"Total de imagens: {total}")
    print(f"Modelo: {model_path}")
    print(f"Saida: {output_dir}")

    device = 0 if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()
    model = YOLO(str(model_path))

    global_idx = 0
    for start in range(0, total, args.batch):
        batch_paths = paths[start : start + args.batch]
        results = model.predict(
            source=batch_paths,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            half=use_half,
            verbose=False,
        )

        for i, result in enumerate(results):
            original_bgr = cv2.imread(batch_paths[i])
            if original_bgr is None:
                raise RuntimeError(f"Falha ao carregar imagem: {batch_paths[i]}")

            annotated = original_bgr.copy()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                for box in boxes_xyxy:
                    x1, y1, x2, y2 = box[:4]
                    p1 = (int(np.floor(x1)), int(np.floor(y1)))
                    p2 = (int(np.ceil(x2)), int(np.ceil(y2)))
                    # Azul em BGR
                    cv2.rectangle(annotated, p1, p2, (255, 0, 0), 2)

            out_name = f"{args.tag}_img{global_idx}.png"
            out_path = output_dir / out_name
            ok = cv2.imwrite(str(out_path), annotated)
            if not ok:
                raise RuntimeError(f"Falha ao salvar imagem: {out_path}")
            global_idx += 1

        done = min(start + args.batch, total)
        print(f"Salvas {done}/{total} imagens")

    print("Concluido.")


if __name__ == "__main__":
    main()
