from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from Arch import NewDirectModel_Inference as NDM
import time

# ================= CONFIG =================
pd.options.display.max_columns = None
USE_KALMAN = False


class KalmanFrame:
    def __init__(self, process_var=1e-1, measurement_var=25.0):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.estimate = None
        self.error = None

    def update(self, frame):
        measurement = frame.astype(np.float32)

        if self.estimate is None or self.estimate.shape != measurement.shape:
            self.estimate = measurement.copy()
            self.error = np.ones_like(measurement, dtype=np.float32)

        # Etapa de previsao
        self.error += self.process_var

        # Etapa de correcao
        kalman_gain = self.error / (self.error + self.measurement_var)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error = (1 - kalman_gain) * self.error

        return np.clip(self.estimate, 0, 255).astype(np.uint8)


def apply_kalman_filter(frame, detection_id, filter_bank):
    if detection_id not in filter_bank:
        filter_bank[detection_id] = KalmanFrame()
    return filter_bank[detection_id].update(frame)

def pad_to_square_center(img):
    h, w = img.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=img.dtype)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = img
    return padded

# ================= MODELOS =================
Segmentation_yolo8s = YOLO("models\\Yolo_v8\\weights\\best.pt")
Segmentation_yolo11s = YOLO("models\\Yolov11s\\weights\\best.pt")
Segmentation_yolo26s = YOLO("models\\Yolo_v26s\\weights\\best.pt")

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

Transform = A.Compose([
            A.Resize(120, 120),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

#Regressor_resnet = NDM("resnet").load_model(r"C:\Users\Clayton\Desktop\MODELS\ResNet-18_120x120.pth")
# Regressor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Regressor = NDM(backbone_name="resnet")
#path = r"C:\Users\Clayton\Desktop\MODELS\ResNet-18_120x120.pth"
#Regressor.load_model(path)


def load_ndm_resnet(path, device):
    model = NDM("resnet")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return model.load_model(path)

    # Fallback para arquivos salvos apenas como state_dict.
    model.load_backbone()
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model


Regressor = load_ndm_resnet(r"models\regressor.pth", device)
Regressor_efficient = NDM("efficientnet_lite").load_model(r"models\EfficientNet-B0_120x120.pth")
Regressor_mobilev3small = NDM("mobilenetv3_small").load_model(r"models\MobileNetV3_Small_120x120.pth")
Regressor_mobilev3large = NDM("mobilenetv3_large").load_model(r"models\MobileNetV3_Large_120x120.pth")


def run_detector_pipeline(detector_name, detector_model, img_full, img_rgb, filter_bank):
    out = detector_model.predict(img_rgb, conf=0.6, verbose=False)[0]
    boxes = out.boxes.xyxy.cpu().numpy().astype(int) if out.boxes is not None else []

    images_raw = []
    detections = []

    for det_id, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if xmax <= xmin or ymax <= ymin:
            continue

        crop = img_full[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        crop_square = pad_to_square_center(crop)
        crop_resized = cv2.resize(crop_square, (120, 120), interpolation=cv2.INTER_LINEAR)
        if USE_KALMAN:
            crop_input = apply_kalman_filter(crop_resized, det_id, filter_bank)
        else:
            crop_input = crop_resized

        images_raw.append(crop_input)
        detections.append({
            "detection_id": det_id,
            "box": (xmin, ymin, xmax, ymax),
        })

    if not images_raw:
        return []

    tensors = [Transform(image=img)["image"] for img in images_raw]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        preds = Regressor(batch)

    pred_resnet = preds.detach().cpu().numpy().flatten().tolist()
    pred_efficient = Regressor_efficient.predict(images_raw)
    pred_mob_small = Regressor_mobilev3small.predict(images_raw)
    pred_mob_large = Regressor_mobilev3large.predict(images_raw)

    for i, detection in enumerate(detections):
        detection["detector_name"] = detector_name
        detection["pred_resnet"] = float(pred_resnet[i])
        detection["pred_efficient"] = float(pred_efficient[i])
        detection["pred_mob_small"] = float(pred_mob_small[i])
        detection["pred_mob_large"] = float(pred_mob_large[i])

    return detections


def draw_detections(frame, frame_id, detections):
    for detection in detections:
        xmin, ymin, xmax, ymax = detection["box"]
        det_id = detection["detection_id"]
        r1 = detection["pred_resnet"]
        r2 = detection["pred_efficient"]
        r3 = detection["pred_mob_small"]
        r4 = detection["pred_mob_large"]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.putText(frame, f"F:{frame_id} ID:{det_id}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"R18:{r1:.2f}", (xmin, ymax + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Eff:{r2:.2f}", (xmin, ymax + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Mv3S:{r3:.2f}", (xmin, ymax + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"Mv3L:{r4:.2f}", (xmin, ymax + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

distance = 60
measuring_tape = 20

print(f"Distância: {distance}cm, Fita métrica: {measuring_tape}cm")
csv_path = f"dataset\\testing\\Oscilation\\Oscilation_D{distance}_F{measuring_tape}_all_detectors.csv"

# ================= WEBCAM =================
cap = cv2.VideoCapture("http://192.168.15.2:8080/video")

if not cap.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam")

print("Executando por 1 minuto...")

# ================= TEMPO =================
start_time = time.time()
EXPOSURE_TIME = 60  

frame_id = 0
results = []

detector_configs = [
    ("yolov8s", Segmentation_yolo8s, True),
    ("yolo11s", Segmentation_yolo11s, False),
    ("yolo26s", Segmentation_yolo26s, False),
]

# Filtro de Kalman separado por detector e por ID de deteccao
kf_banks = {name: {} for name, _, _ in detector_configs}

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

    for detector_name, detector_model, should_draw in detector_configs:
        detections = run_detector_pipeline(
            detector_name,
            detector_model,
            img_full,
            img_rgb,
            kf_banks[detector_name],
        )

        for detection in detections:
            results.append([
                frame_id,
                detection["detector_name"],
                detection["detection_id"],
                measuring_tape,
                detection["pred_resnet"],
                detection["pred_efficient"],
                detection["pred_mob_small"],
                detection["pred_mob_large"],
            ])

        if should_draw:
            draw_detections(frame, frame_id, detections)

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
    "detector",
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
