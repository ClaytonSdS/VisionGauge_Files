from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from Arch import NewDirectModel_Inference as NDM
import os

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = 0

# ======================================================
# FLAGS
# ======================================================
Adjust_Zoom = False
PLOT_RESULTS = False
print("Adjust_Zoom =", Adjust_Zoom)

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
    return cv2.resize(crop_square, (target, target))


# ======================================================
# MODELOS
# ======================================================
Segmentation = YOLO("models/SegARC_v04_lr0.0001_5k/weights/best.pt")

# Regressor_Resnet = NDM("resnet").load_model(r"models\RegArc\Resnet\resnet\014_resnet_120x120_2025_12_12_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")
# Regressor_EF1 = NDM("efficientnet_lite").load_model(r"models\RegArc\Resnet\MIX\efficientnet_lite_120x120_2025_12_01_HashSplit_UnfreezeAll_NoHead_ADAMW.pth")

Regressor_Resnet = NDM("efficientnet_lite").load_model(r"models\RegArc\EfficientNet_678\efficientnet_lite_120x120.pth")
Regressor_EF1 = NDM("efficientnet_lite").load_model(r"models\RegArc\EfficientNet_678\efficientnet_lite_120x120.pth")
Regressor_EF2 = NDM("efficientnet_lite").load_model(r"models\RegArc\EfficientNet_678\efficientnet_lite_120x120.pth")

# ======================================================
# PIPELINE
# ======================================================
paths = [r"dataset\testing\processed\image_test_570.png"]

dataframe = pd.read_csv(r"dataset\testing\dataset_testing_paths.csv")
paths  = dataframe['file'].tolist()

out = Segmentation.predict(paths, conf=0.5)

boxes_list = [out[i].boxes.xyxy.cpu().numpy() for i in range(len(out))]

df_localizer = pd.DataFrame(columns=[
    "boxes", "path",
    "box_xmin", "box_ymin", "box_xmax", "box_ymax",
    "pred_height_cm", "true_height_cm"
])

# ======================================================
# LOOP PRINCIPAL
# ======================================================
paths_count = 0

for boxes in boxes_list:
    n_boxes = boxes.shape[0]

    img_full = cv2.imread(paths[paths_count])
    img_h, img_w = img_full.shape[:2]

    # TRUE LABEL DA IMAGEM
    true_label = dataframe.loc[paths_count, 'deltaH_cm']

    row_indices = []
    images_crop_120 = []

    # -------------------------------
    # Inicializa linhas (1 por bbox)
    # -------------------------------
    for i in range(n_boxes):
        df_localizer.loc[len(df_localizer)] = {
            "boxes": i,
            "path": paths[paths_count],
            "box_xmin": None,
            "box_ymin": None,
            "box_xmax": None,
            "box_ymax": None,
            "pred_height_cm": None,
            "true_height_cm": true_label
        }
        row_indices.append(len(df_localizer) - 1)

    # -------------------------------
    # Crops
    # -------------------------------
    for i in range(n_boxes):
        xmin, ymin, xmax, ymax = boxes[i].astype(int)

        df_localizer.loc[row_indices[i],
            ["box_xmin", "box_ymin", "box_xmax", "box_ymax"]
        ] = [xmin, ymin, xmax, ymax]

        if Adjust_Zoom:
            crop_120 = zoom_out_to_size(img_full, xmin, ymin, xmax, ymax)
        else:
            crop = img_full[ymin:ymax, xmin:xmax]
            crop_120 = pad_to_square_center(crop)

        images_crop_120.append(crop_120)

    # -------------------------------
    # Predições
    # -------------------------------
    preds_resnet = (
        Regressor_Resnet.predict(images_crop_120)
        if n_boxes else np.zeros((0,), float)
    )

    # -------------------------------
    # Salva previsões
    # -------------------------------
    for i in range(n_boxes):
        df_localizer.loc[row_indices[i], "pred_height_cm"] = float(preds_resnet[i])

    paths_count += 1

# ======================================================
# SALVAR CSV
# ======================================================
output_dir = r"dataset\testing"
os.makedirs(output_dir, exist_ok=True)

# abs_error
df_localizer['abs_error'] = abs(df_localizer['pred_height_cm'] - df_localizer['true_height_cm'])            # Erro absoluto para MAE
df_localizer['signed_error'] = (df_localizer['pred_height_cm'] - df_localizer['true_height_cm'])            # Erro para teste de subestimação / superestimação
df_localizer['squared_error'] = df_localizer['signed_error'] ** 2                                           # Erro quadrático para RMSE
df_localizer['relative_error_pct'] = (df_localizer['signed_error'] / df_localizer['true_height_cm']) * 100  # Erro relativo em %


save_path = os.path.join(output_dir, "testing_predictions.csv")
df_localizer.to_csv(save_path, index=False)

print(f"Dataset com as previsões salvo em {save_path}")
