from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
from Arch import NewDirectModel_Inference as NDM
import gc

# ======================================================
# CONFIGURAÇÃO
# ======================================================
image_path = r"steps\inference\visiongauge_inf1.png"
output_dir = r"steps\inference"
os.makedirs(output_dir, exist_ok=True)

Adjust_Zoom = False
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else "cpu"
print("Device:", "CUDA" if use_cuda else "CPU")

# ======================================================
# FUNÇÕES AUXILIARES
# ======================================================
def pad_to_square_center(img):
    h, w = img.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=img.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img
    return padded

def zoom_out_to_size(img, xmin, ymin, xmax, ymax, target=120):
    h, w = img.shape[:2]
    bw = xmax - xmin
    bh = ymax - ymin
    size = max(bw, bh)

    if size >= target:
        crop = img[ymin:ymax, xmin:xmax]
        return cv2.resize(pad_to_square_center(crop), (target, target))

    pad = (target - size) // 2
    xmin, ymin = xmin - pad, ymin - pad
    xmax, ymax = xmax + pad, ymax + pad

    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)

    crop = img[ymin:ymax, xmin:xmax]
    return cv2.resize(pad_to_square_center(crop), (target, target))

# ======================================================
# CARREGAR MODELO YOLO E REGRESSOR
# ======================================================
Segmentation = YOLO("models/SegARC_v04_lr0.0001_5k/weights/best.pt")
Regressor = NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\Resnet\resnet_120x120.pth")

# ======================================================
# PROCESSAR IMAGEM
# ======================================================
img_full = cv2.imread(image_path)
if img_full is None:
    raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

# Fazer predição YOLO
results = Segmentation.predict(
    source=image_path,
    conf=0.5,
    imgsz=640,
    device=device,
    half=use_cuda,
    verbose=False
)

result = results[0]
boxes = result.boxes.xyxy.cpu().numpy()
n_boxes = boxes.shape[0]

if n_boxes == 0:
    print("Nenhum objeto detectado na imagem.")
else:
    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.astype(int)

        # --- Crop para o regressor ---
        if Adjust_Zoom:
            bbox_crop = zoom_out_to_size(img_full, xmin, ymin, xmax, ymax)
        else:
            bbox_crop = pad_to_square_center(img_full[ymin:ymax, xmin:xmax])

        # --- Predição do regressor ---
        pred_height = float(Regressor.predict([bbox_crop])[0])

        # --- Imagem com bounding box e rótulo ---
        img_with_bbox = img_full.copy()
        cv2.rectangle(img_with_bbox, (xmin, ymin), (xmax, ymax), (0, 0, 255), 20)  # espessura da caixa

        # Texto do rótulo
        label_text = f"fluid_height: {pred_height:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 5

        # Calcular tamanho do texto
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Posição do texto
        text_x = xmin
        text_y = max(ymin - 10, text_height + 5)

        # --- Caixa preta atrás do texto ---
        cv2.rectangle(
            img_with_bbox,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + baseline - 5),
            (0, 0, 0),  # cor preta
            -1  # preenchido
        )

        # --- Colocar texto em branco ---
        cv2.putText(
            img_with_bbox,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),  # texto branco
            thickness
        )

        # --- Salvar imagens ---
        cv2.imwrite(os.path.join(output_dir, f"image_with_bbox_{idx}.jpg"), img_with_bbox)
        cv2.imwrite(os.path.join(output_dir, f"bbox_only_{idx}.jpg"), bbox_crop)

        print(f"Box {idx}: Predicted height = {pred_height:.3f}")

    print(f"{n_boxes} bounding box(es) processadas e salvas em {output_dir}")

# Limpeza
del results
torch.cuda.empty_cache()
gc.collect()
