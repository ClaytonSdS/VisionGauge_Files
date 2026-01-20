from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
from Arch import NewDirectModel_Inference as NDM
import time
import matplotlib.pyplot as plt

# Parallax Parameters
n_air = 1.000293 # índice de refração do ar
n_m = 1.53 # índice de refração do acrílico
t = 0.1 # espessura do acrílico em cm 1mm
d = 26 # distância entre o objeto e a lente em cm


# ================= CONFIG =================
USE_SMOOTHING = False  

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



# ---------------- Inicialização dos modelos ----------------
Segmentation = YOLO("models/SegARC_v08/weights/best.pt")

Regressor_Resnet = NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\Resnet\resnet_120x120.pth")
Regressor = NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\Resnet\unfreeze_last2\resnet_120x120_2026_01_11_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")


# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.2.117:8080/video")

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

        crop_square = pad_to_square_center(crop)

        images_raw.append(crop_square)
        valid_boxes.append((xmin, ymin, xmax, ymax))

    # ================= PREDIÇÕES =================
    preds_r1 = []
    preds_vit = []
    hp_k = []
    ht_k = []

    if len(images_raw) > 0:
        preds_r1 = Regressor.predict(images_raw)
        preds_resnet = Regressor_Resnet.predict(images_raw)

        hp = preds_r1
        delta = 0.1 # 1mm de espessura do acrílico em cm
        M = 150 # 150 cm
        alpha = np.arctan((M - hp)/d)
        gama = np.arcsin(n_air/n_m * np.sin(alpha))
        delta_h = delta * np.tan(gama)
        ht = hp - delta_h

    # ================= DESENHO =================
    for k, (xmin, ymin, xmax, ymax) in enumerate(valid_boxes):
        bw = xmax - xmin
        bh = ymax - ymin

        hp_k = float(hp[k])
        ht_k = float(ht[k])

        r1_value = float(preds_r1[k])
        resnet_value = float(preds_resnet[k])

        label = f"hp:{hp_k:.1f} | ht:{ht_k:.1f} | {bw}x{bh}"

        print(label)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame,(xmin, ymin - text_h - 10),(xmin + text_w, ymin),(0, 0, 0),-1)
        cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # ================= OpenCV window =================
    cv2.imshow("Webcam - Altura Estimada", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Encerrando...")
        break

# ================= Cleanup =================
cap.release()
cv2.destroyAllWindows()
