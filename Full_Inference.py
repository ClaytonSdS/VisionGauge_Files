from ultralytics import YOLO
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from Arch import NewDirectModel_Inference as NDM
from Arch import ViT_Inference as ViT

pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
pd.options.display.width = 0

# ======================================================
# FLAG DE ZOOM
# ======================================================
Adjust_Zoom = False 
print("Adjust_Zoom =", Adjust_Zoom)

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
# FUNÇÃO: Zoom-out até o crop ter pelo menos 120×120
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


# ======================================================
# MODELOS
# ======================================================
Segmentation = YOLO("models/SegARC_v04_lr0.0001_5k/weights/best.pt")

Regressor_Resnet = NDM("resnet").load_model("models\\RegArc\\MIX\\023_resnet_120x120_2025_12_12_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")
#Regressor_Resnet = NDM("resnet").load_model("models\\RegArc\MIX\\resnet_120x120_2025_12_10_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")

Regressor_EF1 = NDM("efficientnet_lite").load_model("models\\RegArc\\MIX\\efficientnet_lite_120x120_2025_12_04_HashSplit_UnfreezeAll_NoHead_ADAMW_retrained.pth")
Regressor_EF1 = NDM("resnet").load_model("models\\RegArc\MIX\\017_resnet_120x120_2025_12_12_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")

#Regressor_EF2 = NDM("efficientnet_lite").load_model( "models\\RegArc\\MIX\\efficientnet_lite_120x120_2025_12_04_HashSplit_UnfreezeAll_NoHead_ADAMW_retrained.pth")
Regressor_EF2 = NDM("efficientnet_lite").load_model( "models\RegArc\MIX\efficientnet_lite_120x120_2025_12_10_HashSplit_Unfreeze_NoHead_ADAMW_retrained.pth")

Regressor_ViT = ViT("vit").load_model( "models\\RegArc\\MIX\\vit_b_16_224x224_2025_12_06_HashSplit_Unfreeze_NoHead_ADAMW.pth")

# ======================================================
# PIPELINE
# ======================================================
paths = [
    r"dataset\dataset_od\6k\test\16.6\image_resized_4823_png.rf.8b8157f71c117af388f48cf409655fd9.jpg",
    r"dataset\dataset_od\6k\test\16.6\image_resized_6521_png.rf.0c4c93389eaa085a229b5717400bfb4a.jpg",
    r"dataset\dataset_od\6k\test\11.1\image_resized_4735_png.rf.e4a4842315253a922b6d0e0139e26e19.jpg",
    r"dataset\dataset_od\6k\test\32.3\image_resized_5017_png.rf.081da2d1678e94794db08ac856074646.jpg",
    r"dataset\dataset_od\6k\test\37.8\image_resized_4278_png.rf.c8fa62c6464f964573f9a986dc04313e.jpg",
    r"dataset\dataset_od\6k\valid\37.6\image_resized_1137_png.rf.2684dc2c9c59543af5eecc463f54479f.jpg"
]

out = Segmentation.predict(paths, conf=0.5)

numero_de_imgs = len(out)
boxes_list = [out[i].boxes.xyxy.cpu().numpy() for i in range(numero_de_imgs)]

df_localizer = pd.DataFrame(columns=[
    "boxes", "path", "box_xmin", "box_ymin", "box_xmax", "box_ymax", "pred_height_cm"
])

# ======================================================
# LOOP PRINCIPAL
# ======================================================
paths_count = 0

for boxes in boxes_list:
    n_boxes = boxes.shape[0]

    img_full = cv2.imread(paths[paths_count])
    img_h, img_w = img_full.shape[:2]

    row_indices = []
    images_crop_120 = []

    # inicializa linhas
    for i in range(n_boxes):
        df_localizer.loc[len(df_localizer)] = {
            "boxes": i,
            "path": paths[paths_count],
            "box_xmin": None,
            "box_ymin": None,
            "box_xmax": None,
            "box_ymax": None,
            "pred_height_cm": None
        }
        row_indices.append(len(df_localizer) - 1)

    # crops
    for i in range(n_boxes):
        xmin, ymin, xmax, ymax = boxes[i].astype(int)

        df_localizer.loc[row_indices[i], ["box_xmin","box_ymin","box_xmax","box_ymax"]] = \
            [xmin, ymin, xmax, ymax]

        if Adjust_Zoom:
            crop_120 = zoom_out_to_size(img_full, xmin, ymin, xmax, ymax, target=120)
        else:
            crop = img_full[ymin:ymax, xmin:xmax]
            crop_120 = pad_to_square_center(crop)

        images_crop_120.append(crop_120)

    # ==================================================
    # PREDIÇÕES
    # ==================================================
    preds_resnet = Regressor_Resnet.predict(images_crop_120) if n_boxes else np.zeros((0,), float)
    preds_ef1    = Regressor_EF1.predict(images_crop_120) if n_boxes else np.zeros((0,), float)
    preds_ef2    = Regressor_EF2.predict(images_crop_120) if n_boxes else np.zeros((0,), float)
    preds_vit    = Regressor_ViT.predict(images_crop_120) if n_boxes else np.zeros((0,), float)

    # ==================================================
    # SALVAR SOMENTE RESNET NO DF
    # ==================================================
    for i in range(n_boxes):
        df_localizer.loc[row_indices[i], "pred_height_cm"] = float(preds_resnet[i])

    # ==================================================
    # PLOT 1x4
    # ==================================================
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    for ax in axs:
        ax.imshow(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))
        ax.axis("off")

    axs[0].set_title("ResNet")
    axs[1].set_title("EffNet-Lite 1")
    axs[2].set_title("EffNet-Lite 2")
    axs[3].set_title("ViT")

    bbox_info = []

    for i in range(n_boxes):
        xmin = int(df_localizer.loc[row_indices[i], "box_xmin"])
        ymin = int(df_localizer.loc[row_indices[i], "box_ymin"])
        xmax = int(df_localizer.loc[row_indices[i], "box_xmax"])
        ymax = int(df_localizer.loc[row_indices[i], "box_ymax"])

        bw = xmax - xmin
        bh = ymax - ymin
        bbox_info.append(f"{bw}x{bh}")

        values = [
            float(preds_resnet[i]),
            float(preds_ef1[i]),
            float(preds_ef2[i]),
            float(preds_vit[i])
        ]

        labels = [
            f"ResNet: {values[0]:.2f}",
            f"EF1: {values[1]:.2f}",
            f"EF2: {values[2]:.2f}",
            f"ViT: {values[3]:.2f}"
        ]

        for ax, label in zip(axs, labels):
            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    bw, bh,
                    fill=False,
                    edgecolor='lime',
                    linewidth=2
                )
            )
            ax.text(
                xmin,
                ymin - 6,
                label,
                color='lime',
                fontsize=10,
                backgroundcolor='black'
            )

    # título
    path_parts = paths[paths_count].split("\\")
    folder = path_parts[2].upper() if len(path_parts) > 2 else "FOLDER"
    filename = path_parts[-1]
    bbox_text = " | ".join(bbox_info) if bbox_info else "No boxes"

    plt.suptitle(
        f"{folder} | {filename} | Img: {img_w}x{img_h} | BBoxes: {bbox_text}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    paths_count += 1

print(df_localizer)
