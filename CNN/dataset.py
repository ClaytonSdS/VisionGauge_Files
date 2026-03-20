import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from collections import OrderedDict
import os
import numpy as np
from PIL import Image
import torch
import pandas as pd
import cv2

class CNN_Dataset(Dataset):
    def __init__(self, data:pd.DataFrame, feature_column:str, target_column:str, image_size:tuple=(120,120), 
                 cache_size=50, is_valid:bool=True):

        self.data = data.reset_index(drop=True)
        self.image_size = image_size 
        
        self.cache_size = cache_size
        self._cache = OrderedDict()

        self.feature_column = feature_column
        self.target_column = target_column

        self.height, self.width = image_size if isinstance(image_size, tuple) else (image_size, image_size)

        # Albumentations COMPOSE (criado uma vez)
        if is_valid:
            # TRANSFORMAÇÕES DE VALIDAÇÃO (SEM AUGMENTAÇÃO)
            self.transform = A.Compose([
                A.Resize(self.height, self.width),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # TRANSFORMAÇÕES DE TREINO (COM AUGMENTAÇÃO)
            self.transform = A.Compose([
                A.RandomBrightnessContrast(0.1,0.2,p=0.4),
                A.RGBShift(4,4,4,p=0.3),

                # Zoom In/Out
                A.Affine(scale=[0.9,1.2],translate_percent=[0, 0],rotate=[0, 0],shear=[0, 0], interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                         fit_output=False, keep_ratio=True, rotate_method="ellipse", balanced_scale=True, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
                
                # Blur Addition
                #A.OneOf([A.Defocus(radius=(1, 1.3), alias_blur=(0.01, 0.08), p=0.9), 
                        # A.MotionBlur(blur_limit=(3, 3), allow_shifted=False, p=0.1)], p=0.2),
                
                # Noise Addition
                A.OneOf([A.ISONoise(color_shift=(0.002, 0.01), intensity=(0.03, 0.07), p=0.6), 
                         A.GaussNoise(std_range=(0.03, 0.05), mean_range=(0, 0), per_channel=True, noise_scale_factor=1, p=0.4)], p=0.8),

                A.HueSaturationValue(hue_shift_limit=[-20, 20],sat_shift_limit=[-30, 30],val_shift_limit=[-20, 20], p=0.5),

                A.ImageCompression(compression_type="jpeg", quality_range=(8, 10), p=0.5),

                A.Rotate(limit=[-30, 30],interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT, rotate_method="ellipse",
    crop_border=False,mask_interpolation=cv2.INTER_NEAREST,fill=0,fill_mask=0, p=0.5),

                #A.PlanckianJitter(mode="blackbody",temperature_limit=(4500, 6500),sampling_method="uniform", p=0.2),

                # Aplicar Resize e Normalize Padrão
                A.Resize(self.height, self.width),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.data)

    def __apply_transform__(self, img):
        return self.transform(image=img)["image"]

    def __get_image__(self, idx):
        img_path = self.data.iloc[idx][self.feature_column]
        return np.array(Image.open(img_path).convert("RGB"))

    def __getitem__(self, idx):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            img, y = self._cache[idx]
            return img.clone(), y.clone()

        # Ler a imagem usando Pillow
        img = self.__get_image__(idx)

        # Aplicar transformações, i.e., data augmentation
        img = self.__apply_transform__(img)

        # Target
        y = torch.tensor(self.data.iloc[idx][self.target_column], dtype=torch.float32)

        self._cache[idx] = (img, y)
        self._cache.move_to_end(idx)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        
        return img.clone(), y.clone()
