from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
from Arch import NewDirectModel_Inference as NDM
from Arch import ViT_Inference as ViT
import time
import matplotlib.pyplot as plt

# ================= CONFIG =================
Adjust_Zoom = False
USE_SMOOTHING = False  
USE_PLOT = False

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = 0

# ======================================================
# FUNÇÃO: Zero-padding para tornar a imagem quadrada
# ======================================================
def pad_to_square_center(img):
    h, w = img.shape[:2]
    size = max(h, w)

    padded = np.zeros((size, size, 3), dtype=img.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2

    padded[y_off:y_off + h, x_off:x_off + w] = img
    return padded


# ======================================================
# FUNÇÃO: Zoom-out até o crop ser >= 120×120
# ======================================================
def zoom_out_to_size(img, xmin, ymin, xmax, ymax, target=120):
    h, w = img.shape[:2]

    bw = xmax - xmin
    bh = ymax - ymin
    size = max(bw, bh)

    if size >= target:
        crop = img[ymin:ymax, xmin:xmax]
        square = pad_to_square_center(crop)
        return cv2.resize(square, (target, target))

    needed = target - size
    pad = needed // 2

    new_xmin = xmin - pad
    new_ymin = ymin - pad
    new_xmax = xmax + pad
    new_ymax = ymax + pad

    while (new_xmax - new_xmin) < target or (new_ymax - new_ymin) < target:
        new_xmin -= 1
        new_ymin -= 1
        new_xmax += 1
        new_ymax += 1

    pad_left = max(0, -new_xmin)
    pad_top = max(0, -new_ymin)
    pad_right = max(0, new_xmax - w)
    pad_bottom = max(0, new_ymax - h)

    new_xmin = max(0, new_xmin)
    new_ymin = max(0, new_ymin)
    new_xmax = min(w, new_xmax)
    new_ymax = min(h, new_ymax)

    crop = img[new_ymin:new_ymax, new_xmin:new_xmax]

    crop = cv2.copyMakeBorder(
        crop,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    crop_square = pad_to_square_center(crop)
    crop_final = cv2.resize(crop_square, (target, target))

    return crop_final


# ---------------- Inicialização dos modelos ----------------
Segmentation = YOLO("models/SegARC_v04_lr0.0001_5k/weights/best.pt")

Regressor = NDM("efficientnet_lite").load_model("models\\RegArc\\MIX\\efficientnet_lite_120x120_2025_12_04_HashSplit_UnfreezeAll_NoHead_ADAMW_retrained.pth")

Regressor_ResNet = NDM("resnet").load_model("models\\RegArc\\MIX\\resnet_120x120_2025_12_05_HashSplit_Unfreeze_NoHead_ADAMW.pth")
Regressor_ResNet = NDM("resnet").load_model("models\\RegArc\\MIX\\023_resnet_120x120_2025_12_12_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")

Regressor_ViT = ViT("vit").load_model("models\\RegArc\\MIX\\vit_b_16_224x224_2025_12_07_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")

Regressor_ViT = None


# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("http://192.168.2.117:8080/video")

if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam")

print("Pressione Q para sair")

# ======================================================
# PARAMETROS DO SMOOTHING TEMPORAL
# ======================================================
smooth_boxes = {}
alpha = 0.8


# ---------------- LOOP PRINCIPAL ----------------
while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, (854, 640))
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # rotacionar o frame 90 an
    if not ret:
        print("Frame não capturado.")
        break

    img_full = frame.copy()
    img_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

    # ================= YOLO Prediction =================
    out = Segmentation.predict(img_rgb, conf=0.5, verbose=False)[0]
    boxes = out.boxes.xyxy.cpu().numpy()
    n_boxes = boxes.shape[0]

    # ================= SMOOTHING CONDICIONAL =================
    if USE_SMOOTHING:
        smoothed = []

        for i in range(n_boxes):
            xmin, ymin, xmax, ymax = boxes[i]

            if i in smooth_boxes:
                prev = smooth_boxes[i]
                xmin = alpha * prev[0] + (1 - alpha) * xmin
                ymin = alpha * prev[1] + (1 - alpha) * ymin
                xmax = alpha * prev[2] + (1 - alpha) * xmax
                ymax = alpha * prev[3] + (1 - alpha) * ymax

            smooth_boxes[i] = (xmin, ymin, xmax, ymax)
            smoothed.append([xmin, ymin, xmax, ymax])

        boxes = np.array(smoothed).astype(int)

    else:
        boxes = boxes.astype(int)

    # ================= CROP + PAD =================
    images_raw = []
    valid_boxes = []

    for i in range(n_boxes):
        xmin, ymin, xmax, ymax = boxes[i]
        crop = img_full[ymin:ymax, xmin:xmax]

        if xmax <= xmin or ymax <= ymin:
            continue

        if crop.size == 0:
            continue

        if Adjust_Zoom:
            crop_square = zoom_out_to_size(img_full, xmin, ymin, xmax, ymax, target=120)
        else:
            crop_square = pad_to_square_center(crop)

        images_raw.append(crop_square)
        valid_boxes.append((xmin, ymin, xmax, ymax))

    # ================= PREDIÇÕES =================
    preds_r1 = []
    preds_vit = []

    if len(images_raw) > 0:
        preds_r1 = Regressor.predict(images_raw)

        if Regressor_ViT != None:
            images_vit = [cv2.resize(img, (224, 224)) for img in images_raw]
            preds_vit = Regressor_ViT.predict(images_vit)

        preds_resnet = Regressor_ResNet.predict(images_raw)

    # ================= DESENHO =================
    for k, (xmin, ymin, xmax, ymax) in enumerate(valid_boxes):

        bw = xmax - xmin
        bh = ymax - ymin

        r1_value = float(preds_r1[k]) if len(preds_r1) > k else 0.0
        resnet_value = float(preds_resnet[k]) if len(preds_resnet) > k else 0.0
        vit_value = float(preds_vit[k]) if len(preds_vit) > k else 0.0

        label = f"Resnet:{resnet_value:.1f} | R1:{r1_value:.1f} | ViT:{vit_value:.1f} | {bw}x{bh}"

        print(label)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (xmin, ymin - text_h - 10),
            (xmin + text_w, ymin),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ================= SUBPLOT (3 imagens) =================
    if USE_PLOT:
        if len(valid_boxes) > 0:
            xmin, ymin, xmax, ymax = valid_boxes[0]

            crop_raw = img_full[ymin:ymax, xmin:xmax]
            crop_raw = pad_to_square_center(crop_raw)
            crop_zoom = zoom_out_to_size(img_full, xmin, ymin, xmax, ymax, target=224)
            crop_224 = cv2.resize(crop_raw, (224, 224))

            crop_raw = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2RGB)
            crop_224 = cv2.cvtColor(crop_224, cv2.COLOR_BGR2RGB)
            crop_zoom = cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots(1, 3, figsize=(8, 4))

            ax[0].imshow(crop_raw)
            ax[0].set_title("Crop Original")
            ax[0].axis("off")

            ax[1].imshow(crop_224)
            ax[1].set_title("Crop pad 224x224")
            ax[1].axis("off")

            ax[2].imshow(crop_zoom)
            ax[2].set_title("Crop Zoom 224x224")
            ax[2].axis("off")

            plt.show(block=True)
            plt.close(fig)

    # ================= OpenCV window =================
    cv2.imshow("Webcam - Altura Estimada", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Encerrando...")
        break

# ================= Cleanup =================
cap.release()
cv2.destroyAllWindows()
