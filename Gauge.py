import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import os
import pandas as pd
from collections import OrderedDict
from ultralytics import YOLO
from Arch import NewDirectModel_Inference as NDM
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# Custom Bilateral Filter transform
class BilateralFilter(A.ImageOnlyTransform):
    def __init__(self, diameter=5, sigma_color=30, sigma_space=30, always_apply=False, p=1.0):
        super(BilateralFilter, self).__init__(always_apply, p)
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img, **params):
        return cv2.bilateralFilter(img, self.diameter, self.sigma_color, self.sigma_space)

# Define the transformation pipeline
Transform = A.Compose([
    A.Resize(120, 120),
    BilateralFilter(
        diameter=3,
        sigma_color=30,
        sigma_space=30,
        p=1.0
    ),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Custom Dataset with memory cache control
class CustomDataset(Dataset):
    def __init__(self, data, image_size=(120, 120), DATASET_PATH="dataset", cache_size=50, transform=None):
        self.data = data.reset_index(drop=True)
        self.image_size = image_size
        self.DATASET_PATH = DATASET_PATH
        self.cache_size = cache_size
        self.transform = transform
        self._cache = OrderedDict()
        print(self.data.head())  # Print the first few rows of the data to check
        print("esse é os dados")

    def __len__(self):
        return len(self.data)

    def __get_image__(self, idx):
        img_path = os.path.join(self.data.iloc[idx]["path"])
        return np.array(Image.open(img_path).convert("RGB"))

    def __getitem__(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            img, y = self._cache[idx]
            return img.clone(), y.clone()

        img = self.__get_image__(idx)
        y = torch.tensor(self.data.iloc[idx]["true_height_cm"], dtype=torch.float32)

        if self.transform:
            img = self.transform(image=img)["image"]

        self._cache[idx] = (img, y)
        self._cache.move_to_end(idx)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return img.clone(), y.clone()

# VisionGauge class
class VisionGauge:
    def __init__(self, segmentation_model_path, regressor_model_path, batch_size=16, image_size=(120, 120), dataset_path="dataset", dataframe=None):
        # Load segmentation and regressor models
        self.segmentation = YOLO(segmentation_model_path)
        self.regressor = NDM("efficientnet_lite").load_model(regressor_model_path)

        # Parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dataframe to store true height labels
        self.dataframe = dataframe

    def predict_batch(self, loader):
        self.regressor.eval()
        preds = []
        with torch.no_grad():
            for img, _ in tqdm(loader, desc="Predicting in batches"):
                img = img.to(self.device, non_blocking=True)

                # Batch inference
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    output = self.regressor(img).squeeze(-1)

                preds.append(output.cpu().numpy())

        return np.concatenate(preds)


    def predict_paths(self, image_paths, adjust_zoom=False, plot_results=False):
        # Lista de predições para as imagens
        out = self.segmentation.predict(image_paths, conf=0.5)
        boxes_list = [out[i].boxes.xyxy.cpu().numpy() for i in range(len(out))]

        # DataFrame inicial
        df_localizer = pd.DataFrame(columns=[
            "boxes", "path", "box_xmin", "box_ymin", "box_xmax", "box_ymax", 
            "pred_height_cm", "true_height_cm"
        ])

        # Contador de imagens
        paths_count = 0

        # ======================================================
        # LOOP PRINCIPAL
        # ======================================================
        for boxes in boxes_list:
            n_boxes = boxes.shape[0]
            img_full = cv2.imread(image_paths[paths_count])  # Lê a imagem completa
            img_h, img_w = img_full.shape[:2]

            # Obtém o rótulo verdadeiro para a altura da imagem
            true_label = self.get_true_label(paths_count)

            # -------------------------------
            # Inicializa as linhas (1 por bbox)
            # -------------------------------
            row_indices = []  # Lista para armazenar os índices das linhas
            images_crop_120 = []  # Lista para armazenar as imagens cortadas

            # Para cada box da imagem, vamos adicionar uma linha ao DataFrame
            for i in range(n_boxes):
                df_localizer.loc[len(df_localizer)] = {
                    "boxes": i,
                    "path": image_paths[paths_count],
                    "box_xmin": None,
                    "box_ymin": None,
                    "box_xmax": None,
                    "box_ymax": None,
                    "pred_height_cm": None,
                    "true_height_cm": true_label  # Aplica o rótulo verdadeiro a todos os boxes
                }
                row_indices.append(len(df_localizer) - 1)

            # -------------------------------
            # Processa os crops das caixas
            # -------------------------------
            for i in range(n_boxes):
                xmin, ymin, xmax, ymax = boxes[i].astype(int)

                # Atualiza as coordenadas das caixas no DataFrame
                df_localizer.loc[row_indices[i], ["box_xmin", "box_ymin", "box_xmax", "box_ymax"]] = [xmin, ymin, xmax, ymax]

                # Ajusta o zoom se necessário, ou realiza o crop e padding
                if adjust_zoom:
                    crop_120 = self.zoom_out_to_size(img_full, xmin, ymin, xmax, ymax)
                else:
                    crop = img_full[ymin:ymax, xmin:xmax]
                    crop_120 = self.pad_to_square_center(crop)

                images_crop_120.append(crop_120)

            # Incrementa o contador para a próxima imagem
            paths_count += 1

        # Após o loop principal, você pode realizar as predições em batch e adicionar a altura predita ao DataFrame
        print("Running batch predictions...")
        dataset = CustomDataset(data=df_localizer, image_size=self.image_size, DATASET_PATH=self.dataset_path, transform=Transform)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        preds = self.predict_batch(loader)

        # -------------------------------
        # Atualiza a coluna pred_height_cm
        # -------------------------------
        pred_idx = 0

        print(f"O tamanho do dataframe é: {df_localizer.shape}")
        print(f"O tamanho do dataframe[true_height_cm] é: {len(df_localizer['true_height_cm'])}")
        print(f"O tamanho do dataframe[pred_height_cm] é: {len(df_localizer['pred_height_cm'])}")
        for i in range(len(df_localizer)):
            if pred_idx < len(preds):
                df_localizer.loc[i, "pred_height_cm"] = float(preds[pred_idx])
                pred_idx += 1

        print("Predictions complete.")
        return df_localizer



    def get_true_label(self, path_idx):
        # Access the dataframe to retrieve the true height value
        return self.dataframe.loc[path_idx, 'deltaH_cm']

    def zoom_out_to_size(self, img, xmin, ymin, xmax, ymax, target=120):
        h, w = img.shape[:2]
        bw, bh = xmax - xmin, ymax - ymin
        size = max(bw, bh)

        if size >= target:
            crop = img[ymin:ymax, xmin:xmax]
            return cv2.resize(self.pad_to_square_center(crop), (target, target))

        needed = target - size
        pad = needed // 2
        new_xmin, new_ymin, new_xmax, new_ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad

        while (new_xmax - new_xmin) < target or (new_ymax - new_ymin) < target:
            new_xmin -= 1
            new_ymin -= 1
            new_xmax += 1
            new_ymax += 1

        crop = img[max(0, new_ymin):min(h, new_ymax), max(0, new_xmin):min(w, new_xmax)]
        return cv2.resize(self.pad_to_square_center(crop), (target, target))

    def pad_to_square_center(self, img):
        h, w = img.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=img.dtype)
        y_off, x_off = (size - h) // 2, (size - w) // 2
        padded[y_off:y_off + h, x_off:x_off + w] = img
        return padded
