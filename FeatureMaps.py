from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
from Arch import Regressor, NewDirectModel

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = 0

# ======================================================================
#   MODELOS
# ======================================================================
Segmentation = YOLO("models/SegARC_2k_augmented/weights/best.pt")
Regressor = NewDirectModel(backbone_name="efficientnet_lite").load_model("models\\RegArc\\z.pth")

# ======================================================================
#   FUNÇÃO — RESIZE COM PADDING
# ======================================================================
def resize_with_padding(img, target_size=224, pad_color=(0,0,0)):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded


# ======================================================================
#   HOOKS PARA CAPTURAR AS ATIVAÇÕES DO REGRESSOR
# ======================================================================
activation_maps = {}

def save_activation(name):
    def hook(module, inp, out):
        activation_maps[name] = out.detach().cpu()
    return hook

# Registrar hook em TODAS as camadas Conv2D do Regressor
for name, module in Regressor.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        module.register_forward_hook(save_activation(name))

# ======================================================================
#   FUNÇÃO PARA PLOTAR FEATURE MAPS
# ======================================================================
def plot_feature_maps(layer_name, fmap, max_plots=16):
    fmap = fmap.squeeze(0)  # remove batch -> (C, H, W)
    num_filters = fmap.shape[0]

    n = min(num_filters, max_plots)
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(12, 3 * rows))
    plt.suptitle(f"Camada: {layer_name} — exibindo {n}/{num_filters} filtros")

    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(fmap[i].numpy(), cmap='gray')
        plt.title(f"Filtro {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ======================================================================
#   PROCESSO PRINCIPAL
# ======================================================================

paths = [
    r"dataset\\training\\resized\\image_resized_38.png"
]

out = Segmentation.predict(paths, conf=0.6)

numero_de_boxes = len(out)
boxes = [out[i].boxes.xyxy.cpu().numpy() for i in range(numero_de_boxes)]

df_localizer = pd.DataFrame({
    "boxes": pd.Series(dtype="int"),
    "path": pd.Series(dtype="string"),
    "box_xmin": pd.Series(dtype="float"),
    "box_ymin": pd.Series(dtype="float"),
    "box_xmax": pd.Series(dtype="float"),
    "box_ymax": pd.Series(dtype="float"),
    "pred_height_cm": pd.Series(dtype="float")
})

paths_count = 0

for boxes in boxes:
    print("====================")
    n_boxes_at_current_image = boxes.shape[0]

    df_localizer = pd.concat([df_localizer,
                              pd.DataFrame({
                                  "boxes": [int(i) for i in range(n_boxes_at_current_image)],
                                  "path": [paths[paths_count]]*n_boxes_at_current_image
                              })],
                             ignore_index=True)

    print(boxes.shape)
    images = []

    for current_box in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[current_box]

        idx = len(df_localizer)-n_boxes_at_current_image + current_box
        df_localizer.loc[idx, ["box_xmin", "box_ymin", "box_xmax", "box_ymax"]] = [xmin, ymin, xmax, ymax]

        img = cv2.imread(df_localizer.loc[idx, "path"])
        crop_image = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        final_img = resize_with_padding(crop_image, target_size=224)
        images.append(final_img)

    images_2_tensor = torch.stack([
        torch.from_numpy(img.transpose(2,0,1)).float() for img in images
    ])
    print(f"Shape do tensor: {images_2_tensor.shape}")

    # --------------------------
    # PREDIÇÃO + CAPTURA DE ATIVAÇÕES
    # --------------------------
    preds = Regressor.predict(images_2_tensor)

    df_localizer.loc[
        len(df_localizer)-n_boxes_at_current_image : len(df_localizer)-1,
        "pred_height_cm"
    ] = preds

    print(f"Predictions: {preds}")

    # --------------------------
    # VISUALIZAÇÃO DOS FEATURE MAPS
    # --------------------------
    print("\n=== FEATURE MAPS DO REGRESSOR ===")
    for layer_name, fmap in activation_maps.items():
        print(f"Plotando {layer_name}  shape={fmap.shape}")
        plot_feature_maps(layer_name, fmap, max_plots=16)


    # --------------------------
    # PLOTAGEM DAS CAIXAS + VALOR PREDITO
    # --------------------------
    img_orig = cv2.imread(paths[paths_count])
    img_draw = img_orig.copy()

    for i in range(n_boxes_at_current_image):
        idx = len(df_localizer)-n_boxes_at_current_image + i

        xmin = int(df_localizer.loc[idx, "box_xmin"])
        ymin = int(df_localizer.loc[idx, "box_ymin"])
        xmax = int(df_localizer.loc[idx, "box_xmax"])
        ymax = int(df_localizer.loc[idx, "box_ymax"])

        pred_cm = float(preds[i])

        cv2.rectangle(img_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        label = f"{pred_cm:.1f} cm"
        cv2.putText(img_draw, label, (xmax - 10, ymax - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Detecções + Predições")
    plt.show()

    paths_count += 1

print(df_localizer[["boxes","box_xmin","box_ymin","box_xmax","box_ymax","pred_height_cm"]].head())
