import os
from pathlib import Path
import argparse

import pandas as pd
import torch
from ultralytics import YOLO

# ============================
# DEFAULT CONFIG
# ============================
DEFAULT_MODEL_PATH = r"models\SegARC_v08\weights\best.pt"
DEFAULT_DATASET_TEST_CSV = r"dataset\testing\dataset_testing_paths.csv"  # colunas: file,variation,true_height_cm
DEFAULT_OUTPUT_TAG = "yolo"

DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH_SIZE = 16


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def maybe_save_confusion_png(tp: int, fn: int, fp: int, tn: int, save_path: Path, title: str) -> bool:
    """Tenta salvar figura da matriz de confusão; retorna False se matplotlib não estiver disponível."""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    # Matriz com linhas=Previsto, colunas=Verdadeiro
    matrix = np.array([[tp, fp], [fn, tn]], dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["fluid-interface", "background"])
    ax.set_yticklabels(["fluid-interface", "background"])
    ax.set_xlabel("Verdadeiro")
    ax.set_ylabel("Previsto")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera matriz de confusao e plot para um modelo YOLO.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Caminho para o arquivo .pt do modelo")
    parser.add_argument("--csv", default=DEFAULT_DATASET_TEST_CSV, help="CSV com coluna 'file'")
    parser.add_argument("--tag", default=DEFAULT_OUTPUT_TAG, help="Prefixo para arquivos de saida")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Tamanho de inferencia")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--output_dir", default=r"dataset\testing\Resultados Detetor", help="Diretorio de saida")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    model_file = (root_dir / args.model).resolve()
    csv_file = (root_dir / args.csv).resolve()

    if not model_file.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {model_file}")
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV de teste nao encontrado: {csv_file}")

    df = pd.read_csv(csv_file)
    if "file" not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'file'.")

    paths = []
    for raw_path in df["file"].astype(str).tolist():
        p = Path(raw_path)
        if not p.is_absolute():
            p = (root_dir / p).resolve()
        paths.append(str(p))

    total_images = len(paths)
    if total_images == 0:
        raise ValueError("Nenhuma imagem encontrada na coluna 'file'.")

    print(f"Total de imagens no CSV: {total_images}")
    print(f"Carregando modelo: {model_file}")

    device = 0 if torch.cuda.is_available() else "cpu"
    use_half = torch.cuda.is_available()
    model = YOLO(str(model_file))

    # Contagem para matriz de confusão (task binaria de presenca de objeto)
    # Regra simplificada por imagem:
    # - Se detectar >=1 caixa: 1 TP.
    # - Se detectar 0 caixas: 1 FN.
    # - FP e TN ficam 0 porque o CSV contem apenas imagens positivas.
    tp = 0
    fn = 0
    fp = 0
    tn = 0

    rows = []

    for start in range(0, total_images, args.batch):
        batch_paths = paths[start:start + args.batch]

        results = model.predict(
            source=batch_paths,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            half=use_half,
            verbose=False,
        )

        for i, result in enumerate(results):
            path = batch_paths[i]
            n_boxes = 0
            if result.boxes is not None:
                n_boxes = int(result.boxes.xyxy.shape[0])

            if n_boxes > 0:
                tp += 1
            else:
                fn += 1

            rows.append(
                {
                    "file": path,
                    "pred_boxes": n_boxes,
                    "has_detection": int(n_boxes > 0),
                    "is_exactly_one_box": int(n_boxes == 1),
                }
            )

        processed = min(start + args.batch, total_images)
        print(f"Processadas {processed}/{total_images} imagens")

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1_score = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)

    # Metricas complementares por imagem
    detected_images = sum(r["has_detection"] for r in rows)
    strict_one_box_images = sum(r["is_exactly_one_box"] for r in rows)
    image_detection_rate = safe_div(detected_images, total_images)
    image_strict_accuracy = safe_div(strict_one_box_images, total_images)

    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    details_csv = output_dir / f"{args.tag}_detection_details.csv"
    metrics_txt = output_dir / f"{args.tag}_metrics.txt"
    cm_csv = output_dir / f"{args.tag}_confusion_matrix.csv"
    cm_png = output_dir / f"{args.tag}_confusion_matrix.png"

    pd.DataFrame(rows).to_csv(details_csv, index=False)

    cm_df = pd.DataFrame(
        [[tp, fp], [fn, tn]],
        index=["Previsto_fluid-interface", "Previsto_background"],
        columns=["Verdadeiro_fluid-interface", "Verdadeiro_background"],
    )
    cm_df.to_csv(cm_csv)

    png_saved = maybe_save_confusion_png(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        save_path=cm_png,
        title=f"Matriz de Confusao - {args.tag}",
    )

    report_lines = [
        f"YOLO - Avaliacao por CSV (coluna file) [{args.tag}]",
        f"model_path: {model_file}",
        f"csv_path: {csv_file}",
        "",
        "Matriz de Confusao (linhas=Previsto, colunas=Verdadeiro)",
        f"TP: {tp}",
        f"FN: {fn}",
        f"FP: {fp}",
        f"TN: {tn}",
        "",
        "Metricas principais",
        f"precision: {precision:.6f}",
        f"recall: {recall:.6f}",
        f"accuracy: {accuracy:.6f}",
        f"f1_score: {f1_score:.6f}",
        "",
        "Metricas complementares por imagem",
        f"image_detection_rate: {image_detection_rate:.6f}",
        f"image_strict_accuracy_(1_box): {image_strict_accuracy:.6f}",
        "",
        "Observacao:",
        "- O CSV contem apenas imagens positivas (com objeto).",
        "- Por isso TN e FP sao 0 neste protocolo simplificado.",
    ]

    metrics_txt.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n===== RESULTADOS =====")
    print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"F1-score:  {f1_score:.6f}")
    print("")
    print(f"Detalhes por imagem: {details_csv}")
    print(f"Matriz (CSV):        {cm_csv}")
    if png_saved:
        print(f"Matriz (PNG):        {cm_png}")
    else:
        print("Matriz (PNG):        nao salva (matplotlib indisponivel)")
    print(f"Relatorio metricas:  {metrics_txt}")


if __name__ == "__main__":
    main()