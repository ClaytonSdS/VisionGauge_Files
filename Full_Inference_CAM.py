from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
from Arch import NewDirectModel_Inference as NDM
import time
import matplotlib.pyplot as plt



# ================= CONFIG =================
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
Regressor = NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\ResNet-18_120x120.pth")

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

    # rotacionar o frame 90 an
    if not ret:
        print("Frame não capturado.")
        break

    img_full = frame.copy()
    img_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

    # ================= YOLO Prediction =================
    out = Segmentation.predict(img_rgb, conf=0.6, verbose=False)[0]
    boxes = out.boxes.xyxy.cpu().numpy()
    n_boxes = boxes.shape[0]

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

    if len(images_raw) > 0:
        preds_r1 = Regressor.predict(images_raw)

    # ================= DESENHO =================
    for k, (xmin, ymin, xmax, ymax) in enumerate(valid_boxes):
        bw = xmax - xmin
        bh = ymax - ymin



        h_p = float(preds_r1[k]) # altura manometrica prevista pelo modelo

        # ===================================================================
        # Parallax Parameters
        n_air = 1.000293 # índice de refração do ar
        n_mangueira = 1.53 # índice de refração do acrílico
        epislon  = 0.1 # espessura do acrílico em cm => 1mm
        d = 29.5 # distância entre o objeto e a lente em cm
        h_c = 22 # altura do centro da lente em relação ao chão

        def delta(L, d):
            alpha = np.arctan(L / d)
            beta =  np.arcsin(n_air/n_mangueira * np.sin(alpha))

            return epislon * np.tan(beta)

        # Cálculo do alpha
        delta_ = delta(L=h_p, d=d)
        h_t = h_p - abs(delta_)

        y = frame.shape[0]
        densidade_pixel = h_p / (ymax - ymin)
        l1 = densidade_pixel * ymin
        delta1 = abs(delta(L=l1, d=d))


        densidade_pixel2 = h_p / (640 - ymin)
        l2 = densidade_pixel2 * ymin
        delta2 = abs(delta(L=l2, d=d))


        # ===================================================================

        label = f"hp:{h_p:.2f} | ht:{h_t:.2f} | {bw}x{bh}"

        print(f"{label}  deltah = {delta_:.2f}cm |, delta1 = {delta1:.2f}cm | delta2 = {delta2:.2f}cm |",
              f"y_min = {ymin} | y_max = {ymax} | y = {y} | d1 = {densidade_pixel:.4f} cm/pixel | d2 = {densidade_pixel2:.4f} cm/pixel | l1 = {l1:.2f} cm | l2 = {l2:.2f} cm")

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
