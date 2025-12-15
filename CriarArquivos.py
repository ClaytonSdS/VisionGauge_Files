import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import shutil

# ---------------- Inicialização do modelo ----------------
Segmentation = YOLO("models/SegARC_2k_augmented/weights/best.pt")

# ---------------- Função de preprocessamento ----------------
def preprocess_for_model(img, image_size=(224, 224)):
    if img.shape[2] == 3 and img.dtype == np.uint8:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = image_size[0] / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_top = (image_size[0] - new_h) // 2
    pad_bottom = image_size[0] - new_h - pad_top
    pad_left = (image_size[1] - new_w) // 2
    pad_right = image_size[1] - new_w - pad_left
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)
    tensor = torch.from_numpy(padded.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0) / 255.0
    return tensor

# ---------------- Caminhos ----------------
dataset_path = r"dataset\dataset_regressor\train"
output_path = os.path.join(dataset_path, "Regressor")

# Remove pasta de saída se já existir (opcional)
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

# ---------------- Loop pelas pastas ----------------
for root, dirs, files in os.walk(dataset_path):
    # Ignora a pasta de saída
    if "Regressor" in root:
        continue

    # Cria a mesma estrutura de pastas no output_path
    relative_path = os.path.relpath(root, dataset_path)
    output_dir = os.path.join(output_path, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    # Processa imagens
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(image_files, desc=f"Processando {relative_path}"):
        img_path = os.path.join(root, img_file)
        out = Segmentation.predict([img_path], conf=0.6)
        
        if len(out) == 0 or len(out[0].boxes) == 0:
            continue

        boxes = out[0].boxes.xyxy.cpu().numpy()
        img = cv2.imread(img_path)

        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            xmin_r, ymin_r, xmax_r, ymax_r = map(lambda x: int(round(x)), [xmin, ymin, xmax, ymax])
            crop = img[ymin_r:ymax_r, xmin_r:xmax_r]

            # Salva o crop na pasta correspondente
            crop_filename = f"{os.path.splitext(img_file)[0]}_box{i}.png"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)