from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from Arch import NewDirectModel_Inference as NDM
import time

# ================= CONFIG =================
pd.options.display.max_columns = None

def pad_to_square_center(img):
    h, w = img.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=img.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img
    return padded

# ================= MODELOS =================
Segmentation = YOLO("models/SegARC_v08/weights/best.pt")

Regressor_resnet = NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\ResNet-18_120x120.pth")
Regressor_efficient = NDM("efficientnet_lite").load_model(r"C:\Users\Clayton\Desktop\MODELS\EfficientNet-B0_120x120.pth")
Regressor_mobilev3small = NDM("mobilenetv3_small").load_model(r"C:\Users\Clayton\Desktop\MODELS\MobileNetV3_Large_120x120.pth")
Regressor_mobilev3large = NDM("mobilenetv3_large").load_model(r"C:\Users\Clayton\Desktop\MODELS\MobileNetV3_Large_120x120.pth")

distance = 20
measuring_tape = 20
csv_path = f"dataset\\testing\\Oscilation\\Oscilation_D{distance}_F{measuring_tape}.csv"

# ================= WEBCAM =================
cap = cv2.VideoCapture("http://192.168.2.117:8080/video")
if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam")

print("Executando por 1 minuto...")

# ================= TEMPO =================
start_time = time.time()
EXPOSURE_TIME = 60  

frame_id = 0
results = []

# ================= LOOP =================
while True:
    elapsed_time = time.time() - start_time
    if elapsed_time >= EXPOSURE_TIME:
        print("1 minuto atingido. Encerrando captura...")
        break

    ret, frame = cap.read()
    if not ret:
        print("Frame não capturado.")
        break

    frame = cv2.resize(frame, (854, 640))
    img_full = frame.copy()
    img_rgb = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)

    # ================= YOLO =================
    out = Segmentation.predict(img_rgb, conf=0.6, verbose=False)[0]
    boxes = out.boxes.xyxy.cpu().numpy().astype(int) if out.boxes is not None else []

    images_raw = []
    valid_ids = []

    for det_id, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if xmax <= xmin or ymax <= ymin:
            continue
        crop = img_full[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue
        crop_square = pad_to_square_center(crop)
        images_raw.append(crop_square)
        valid_ids.append(det_id)

    # Inicializa previsões vazias (evita erro se não houver detecção)
    pred_resnet = pred_efficient = pred_mob_small = pred_mob_large = []

    # ================= REGRESSORES =================
    if len(images_raw) > 0:
        pred_resnet = Regressor_resnet.predict(images_raw)
        pred_efficient = Regressor_efficient.predict(images_raw)
        pred_mob_small = Regressor_mobilev3small.predict(images_raw)
        pred_mob_large = Regressor_mobilev3large.predict(images_raw)

        for i in range(len(images_raw)):
            results.append([
                frame_id,
                valid_ids[i],
                measuring_tape,
                float(pred_resnet[i]),
                float(pred_efficient[i]),
                float(pred_mob_small[i]),
                float(pred_mob_large[i])
            ])

    # ================= DESENHO =================
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if i not in valid_ids:
            continue

        idx = valid_ids.index(i)

        r1 = float(pred_resnet[idx])
        r2 = float(pred_efficient[idx])
        r3 = float(pred_mob_small[idx])
        r4 = float(pred_mob_large[idx])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.putText(frame, f"F:{frame_id} ID:{i}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"R18:{r1:.2f}", (xmin, ymax + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Eff:{r2:.2f}", (xmin, ymax + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Mv3S:{r3:.2f}", (xmin, ymax + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Mv3L:{r4:.2f}", (xmin, ymax + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Frame counter
    cv2.putText(frame, f"Frame: {frame_id}", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    # Cronômetro
    remaining = max(0, EXPOSURE_TIME - elapsed_time)
    cv2.putText(frame, f"Tempo restante: {remaining:05.1f}s",
                (20, 80),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 2)

    cv2.imshow("YOLO + Regressors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

# ================= FINALIZAÇÃO =================
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(results, columns=[
    "frame",
    "detection_id",
    "true_height",
    "ResNet-18",
    "EfficientNet-B0",
    "MobileNetV3 Small",
    "MobileNetV3 Large"
])

df.to_csv(csv_path, index=False)

print(f"\nCSV salvo em: {csv_path}")
print(df.head())
