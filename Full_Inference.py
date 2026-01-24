from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
from collections import OrderedDict
from Arch import NewDirectModel_Inference as NDM
import os
import gc

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = 0

# ======================================================
# FLAGS
# ======================================================
Adjust_Zoom = False
BATCH_SIZE = 8
print("Adjust_Zoom =", Adjust_Zoom)

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
# CACHE LRU
# ======================================================
class ImageCache:
    def __init__(self, max_size=30):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path].copy()

        img = cv2.imread(path)
        self.cache[path] = img

        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return img.copy()


class CropCache:
    def __init__(self, max_size=500):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, img, bbox, adjust_zoom):
        key = (*bbox, adjust_zoom)

        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()

        xmin, ymin, xmax, ymax = bbox

        if adjust_zoom:
            crop = zoom_out_to_size(img, xmin, ymin, xmax, ymax)
        else:
            crop = pad_to_square_center(img[ymin:ymax, xmin:xmax])

        self.cache[key] = crop

        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return crop.copy()


class PredictionCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, model_name, crop, predict_fn):
        key = (model_name, hash(crop.tobytes()))

        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        pred = predict_fn(crop)
        self.cache[key] = pred

        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return pred


# ======================================================
# MODELOS
# ======================================================
output_dir = r"dataset\testing"
os.makedirs(output_dir, exist_ok=True)


Segmentation = YOLO("models/SegARC_v08/weights/best.pt")
Regressor_Resnet = NDM("mobilenetv3_large").load_model(r"C:\Users\Clayton\Desktop\MODELS\mobilenet_v3\LARGE_L3_H0\MobileNetV3_Large_120x120_2.pth")
save_path = os.path.join(output_dir, "MobileNetV3_TESTE2_Predictions.csv")

# ======================================================
# PIPELINE
# ======================================================
dataframe = pd.read_csv(r"dataset\testing\dataset_testing_paths.csv")
paths = dataframe["file"].tolist()

df_localizer = pd.DataFrame(columns=[
    "path", "variation",
    "pred_height_cm", "true_height_cm",
    "n_boxes"
])

image_cache = ImageCache()
crop_cache = CropCache()
prediction_cache = PredictionCache()

paths_count = 0

# ======================================================
# LOOP PRINCIPAL (YOLO EM BATCH)
# ======================================================
for i in range(0, len(paths), BATCH_SIZE):
    batch_paths = paths[i:i + BATCH_SIZE]

    results = Segmentation.predict(
        source=batch_paths,
        conf=0.5,
        imgsz=640,
        device=device,
        half=use_cuda,
        verbose=False
    )

    for result in results:
        path = paths[paths_count]
        img_full = image_cache.get(path)

        boxes = result.boxes.xyxy.cpu().numpy()
        n_boxes = boxes.shape[0]

        true_label = dataframe.loc[paths_count, "true_height_cm"]
        variation = dataframe.loc[paths_count, "variation"]

        # -------------------------------------------------
        # SEM DETECÇÃO → IGNORA
        # -------------------------------------------------
        if n_boxes == 0:
            print(f"Image {paths_count + 1}/{len(paths)} - SEM DETECÇÃO")
            paths_count += 1
            continue

        # -------------------------------------------------
        # CROPS
        # -------------------------------------------------
        preds = []

        for b in range(n_boxes):
            xmin, ymin, xmax, ymax = boxes[b].astype(int)

            crop = crop_cache.get(
                img_full,
                (xmin, ymin, xmax, ymax),
                Adjust_Zoom
            )

            pred = prediction_cache.get(
                model_name="resnet",
                crop=crop,
                predict_fn=lambda x: float(Regressor_Resnet.predict([x])[0])
            )
            preds.append(pred)

        # -------------------------------------------------
        # AGREGAÇÃO (MÉDIA)
        # -------------------------------------------------
        pred_final = float(np.mean(preds))

        df_localizer.loc[len(df_localizer)] = {
            "path": path,
            "variation": variation,
            "pred_height_cm": pred_final,
            "true_height_cm": true_label,
            "n_boxes": n_boxes
        }

        print(
            f"Image {paths_count + 1}/{len(paths)} - "
            f"{n_boxes} box(es) → média = {pred_final:.2f}"
        )

        paths_count += 1
        del result

    del results
    torch.cuda.empty_cache()
    gc.collect()

# ======================================================
# SALVAR CSV
# ======================================================

df_localizer["abs_error"] = abs(
    df_localizer["pred_height_cm"] - df_localizer["true_height_cm"]
)
df_localizer["signed_error"] = (
    df_localizer["pred_height_cm"] - df_localizer["true_height_cm"]
)
df_localizer["squared_error"] = df_localizer["signed_error"] ** 2
df_localizer["relative_error_pct"] = (
    df_localizer["signed_error"] / df_localizer["true_height_cm"]
) * 100


df_localizer.to_csv(save_path, index=False)

print(f"\n✅ Dataset salvo em: {save_path}")
print(f"Imagens processadas com detecção: {len(df_localizer)}")
