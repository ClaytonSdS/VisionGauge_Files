from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
from collections import OrderedDict
import os
import gc
import time
import tensorflow as tf

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = 0

# ======================================================
# FLAGS
# ======================================================
Adjust_Zoom = False
BATCH_SIZE = 8
print("Adjust_Zoom =", Adjust_Zoom)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
use_cuda = False
device = "cpu"
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
# CACHE
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

interpreter = tf.lite.Interpreter(
    model_path=r"C:\Users\Clayton\Desktop\MODELS\model_float8.tflite",
    experimental_delegates=[]
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("Input dtype:", input_details[0]['dtype'])
print("Output dtype:", output_details[0]['dtype'])

for tensor in interpreter.get_tensor_details():
    print(tensor['name'], tensor['dtype'], tensor['quantization'])

# ======================================================
# FUNÇÃO DE PREDIÇÃO TFLITE
# ======================================================
def predict_tflite(crop):
    img = cv2.resize(crop, (120, 120))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float32
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std

    img = np.expand_dims(img, axis=0)

    # Quantization input if needed
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        img = img / scale + zero_point
        img = img.astype(np.uint8)

    else:
        img = img.astype(np.float32)  

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output if needed
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return float(output[0][0])

save_path = os.path.join(output_dir, "ResNet_18_TFLite_8bit.csv")

# ======================================================
# PIPELINE
# ======================================================
dataframe = pd.read_csv(r"dataset\testing\dataset_testing_paths.csv")
paths = dataframe["file"].tolist()
print(len(dataframe))

df_localizer = pd.DataFrame(columns=[
    "path", "variation",
    "pred_height_cm", "true_height_cm",
    "n_boxes",
    "regressor_inference_time_ms"
])

image_cache = ImageCache()
crop_cache = CropCache()
prediction_cache = PredictionCache()

paths_count = 0

# ======================================================
# LOOP PRINCIPAL
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

        if n_boxes == 0:
            print(f"Image {paths_count + 1}/{len(paths)} - SEM DETECÇÃO")
            paths_count += 1
            continue

        preds = []
        regressor_time = 0

        for b in range(n_boxes):
            xmin, ymin, xmax, ymax = boxes[b].astype(int)

            crop = crop_cache.get(
                img_full,
                (xmin, ymin, xmax, ymax),
                Adjust_Zoom
            )

            start = time.perf_counter()

            pred = prediction_cache.get(
                model_name="tflite",
                crop=crop,
                predict_fn=predict_tflite
            )

            end = time.perf_counter()
            regressor_time += (end - start)

            preds.append(pred)

        pred_final = float(np.mean(preds))

        df_localizer.loc[len(df_localizer)] = {
            "path": path,
            "variation": variation,
            "pred_height_cm": pred_final,
            "true_height_cm": true_label,
            "n_boxes": n_boxes,
            "regressor_inference_time_ms": regressor_time * 1000
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

print(
    "Tempo médio do regressor:",
    df_localizer["regressor_inference_time_ms"].mean(),
    "ms"
)

print(f"\n✅ Dataset salvo em: {save_path}")
print(f"Imagens processadas com detecção: {len(df_localizer)}")