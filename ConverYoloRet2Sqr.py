import os
import shutil

src_root = r"dataset\dataset_od\5k_FID"
dst_root = r"dataset\dataset_od\5k_FID_SQUARE"

# cria estrutura de destino
splits = ["train", "valid", "test"]

for split in splits:
    os.makedirs(os.path.join(dst_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_root, split, "labels"), exist_ok=True)


def convert_label_to_square(label_path, save_path):
    """Lê um arquivo YOLO, transforma caixas em quadrado e salva."""
    lines_out = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, cx, cy, w, h = parts
            cx = float(cx)
            cy = float(cy)
            w = float(w)
            h = float(h)

            # transforma em quadrado
            side = max(w, h)

            new_line = f"{cls} {cx:.6f} {cy:.6f} {side:.6f} {side:.6f}"
            lines_out.append(new_line)

    with open(save_path, "w") as f:
        f.write("\n".join(lines_out))

import tqdm

for split in tqdm.tqdm(splits):

    src_img_dir = os.path.join(src_root, split, "images")
    src_lbl_dir = os.path.join(src_root, split, "labels")

    dst_img_dir = os.path.join(dst_root, split, "images")
    dst_lbl_dir = os.path.join(dst_root, split, "labels")

    print(f"Processando split: {split}")

    # COPIAR IMAGENS
    for fname in tqdm.tqdm(os.listdir(src_img_dir)):
        src_img = os.path.join(src_img_dir, fname)
        dst_img = os.path.join(dst_img_dir, fname)
        shutil.copy2(src_img, dst_img)

    # CONVERTER LABELS
    for fname in tqdm.tqdm(os.listdir(src_lbl_dir)):
        if not fname.endswith(".txt"):
            continue

        src_lbl = os.path.join(src_lbl_dir, fname)
        dst_lbl = os.path.join(dst_lbl_dir, fname)

        convert_label_to_square(src_lbl, dst_lbl)

print("\nFinalizado! Dataset quadrado salvo em:")
print(dst_root)
