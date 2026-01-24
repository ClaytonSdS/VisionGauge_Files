# Arch.py
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import os
import torch
from torch import nn
from datetime import datetime
from torchvision import models

import os
import torch
from torch import nn
from datetime import datetime
from torchvision import models
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2   

import torchvision.transforms.functional as F
import albumentations as A
import cv2


class BilateralFilter(A.ImageOnlyTransform):
    def __init__(self, diameter=5, sigma_color=30, sigma_space=30, always_apply=False, p=1.0):
        super(BilateralFilter, self).__init__(always_apply, p)
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def apply(self, img, **params):
        return cv2.bilateralFilter(img, self.diameter, self.sigma_color, self.sigma_space)


Transform = A.Compose([
            #A.CLAHE(clip_limit=(2, 2), p=1.0),
            #A.CLAHE(clip_limit=(2, 2), always_apply=True, p=1.0),
            A.Resize(120, 120),
            #A.MedianBlur(blur_limit=3, p=1.0),
            #BilateralFilter(diameter=3,sigma_color=30,sigma_space=30, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])



class NewDirectModel_Inference(nn.Module):
    def __init__(self, backbone_name: str, transform = Transform, input_dim: int = 3, unfreeze_all: bool = False, debug: bool = False):
        super().__init__()
        self.model_name = backbone_name
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug

        self.use_head = False 
        self.unfreeze_all = unfreeze_all
        self.image_size = (224, 224)

        # Transformações (determinísticas)
        self.transform = transform

        self.to(self.device)


    # ---------------------------------------------------------
    # HEAD MLP
    # ---------------------------------------------------------
    def build_head(self, output_features):
        self.head = nn.Sequential(
            nn.Linear(output_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )


    # ---------------------------------------------------------
    # FORWARD
    # ---------------------------------------------------------
    def forward(self, x):
        x = x.to(self.device)

        if hasattr(self, "head") and self.head is not None:
            feats = self.backbone(x)
            return self.head(feats)

        return self.backbone(x)



    # ---------------------------------------------------------
    # LOAD MODEL COMPLETO
    # ---------------------------------------------------------
    def load_model(self, path_or_ckpt):
        if isinstance(path_or_ckpt, str):
            self.ckpt = torch.load(path_or_ckpt, map_location=self.device)
        else:
            self.ckpt = path_or_ckpt

        meta = self.ckpt.get("metadata", {})
        self.model_name = meta.get("backbone_name", self.model_name)
        self.use_head = meta.get("use_head", False)

        # Recria a arquitetura exatamente igual do treino
        self.load_backbone()

        # Carrega pesos
        state = self.ckpt.get("model_state", None)
        if state is None:
            raise ValueError("Checkpoint sem 'model_state'")

        self.load_state_dict(state, strict=False)

        self.to(self.device)
        self.eval()              # desativa dropout + batchnorm
        self.backbone.eval()     # garante que o backbone está em eval
        # ---------------------------------
        return self


    # ---------------------------------------------------------
    # CONSTRUIR BACKBONE
    # ---------------------------------------------------------
    def load_backbone(self):
        name = self.model_name.lower()

        # ------------------- RESNET18 -------------------
        if name in ("resnet", "resnet18"):
            m = models.resnet18(weights=None)

            out_feats = m.fc.in_features

            if self.use_head:
                m.fc = nn.Identity()
                self.build_head(out_feats)
            else:
                m.fc = nn.Linear(out_feats, 1)

            self.backbone = m

        # ------------------- EFFICIENTNET -------------------
        elif name in ("efficientnet_lite", "efficientnet_b0"):
            m = models.efficientnet_b0(weights=None)

            out_feats = m.classifier[1].in_features

            if self.use_head:
                m.classifier[1] = nn.Identity()
                self.build_head(out_feats)
            else:
                m.classifier[1] = nn.Linear(out_feats, 1)

            self.backbone = m

        elif name in ("mobilenetv3_large", "mobilenet_v3_large"):
                m = models.mobilenet_v3_large(weights=None)
                out_feats = m.classifier[3].in_features
            
                # Usar head MLP para regressão
                if self.use_head:
                    m.classifier[3] = nn.Identity()
                    self.build_head(out_feats)

                else:
                    m.classifier[3] = nn.Linear(out_feats, 1)
                self.backbone = m

        elif name in ("mobilenetv3_small"):
                m = models.mobilenet_v3_small(weights=None)
                out_feats = m.classifier[3].in_features
            
                # Usar head MLP para regressão
                if self.use_head:
                    m.classifier[3] = nn.Identity()
                    self.build_head(out_feats)

                else:
                    m.classifier[3] = nn.Linear(out_feats, 1)
                self.backbone = m

        else:
            raise ValueError(f"Backbone '{self.model_name}' inválido.")



    # ---------------------------------------------------------
    # PREDICT EM BATCH
    # ---------------------------------------------------------
    def predict(self, imgs):
        apply_p_smoothing = False

        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        batch = []

        for img in imgs:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self.transform(image=img_rgb)["image"]
            batch.append(tensor)

        batch = torch.stack(batch).to(self.device)

        self.eval()
        self.backbone.eval()

        with torch.no_grad():
            preds_r1 = self.forward(batch).squeeze(1).cpu().numpy()

        # =======================================================
        # APLICANDO EMA - Suavização Temporal
        # =======================================================
        # Se for batch > 1, aplico suavização em cada item
        if apply_p_smoothing:
            if preds_r1.ndim == 1:

                smoothed = []
                for p in preds_r1:

                    if not hasattr(self, "prev_height"):
                        self.prev_height = p   # inicializa

                    p_smooth = 0.7 * self.prev_height + 0.3 * p
                    self.prev_height = p_smooth

                    smoothed.append(p_smooth)

                preds_r1 = np.array(smoothed)

            else:
                # Caso raro: single-value array
                p = preds_r1.item()
                if not hasattr(self, "prev_height"):
                    self.prev_height = p
                preds_r1 = 0.7 * self.prev_height + 0.3 * p
                self.prev_height = preds_r1
                preds_r1 = np.array([preds_r1])
            # =======================================================

        return preds_r1

    
    def predict_paths(self, img):
        """
        Prediz uma única imagem crua (BGR ou RGB).
        Evita empilhamento de batch e funciona para webcam.
        """

        # Se vier BGR do OpenCV → converter para RGB
        if img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        # Aplicar transform (Albumentations → Tensor CHW)
        tensor = self.transform(image=img_rgb)["image"].unsqueeze(0).to(self.device)

        self.eval()
        self.backbone.eval()

        with torch.no_grad():
            pred = self.forward(tensor).item()

        return pred



    