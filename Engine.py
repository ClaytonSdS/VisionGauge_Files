import os, time, copy, random
from collections import OrderedDict
from datetime import datetime
import numpy as np
import pandas as pd
import albumentations as A
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as T
from tqdm.auto import tqdm
from ultralytics import YOLO


# ============================================================
#  YOLOv8 + Regressão
# ============================================================
class YOLOv8_Det_Regression(nn.Module):
    def __init__(self, base_model='yolov8n.pt', num_regression=1, hook_module_idx=-2, input_size=640, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.input_size = input_size
        self.num_regression = num_regression
        self.hook_module_idx = hook_module_idx

        # Carrega modelo YOLO base
        self.yolo_model = YOLO(base_model).model
        modules = list(self.yolo_model.model)

        if self.verbose:
            print("\n🧩 Estrutura YOLOv8 detectada:")
            for i, m in enumerate(modules):
                print(f"[{i}] {m.__class__.__name__}")

        # Detecta canais de saída
        ch = self._infer_channels_by_hook()
        print(f"🔍 Canais detectados no feature map do módulo {hook_module_idx}: {ch}")

        # Cabeça de regressão
        self.reg_head = nn.Sequential(
            nn.Conv2d(ch, 256, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_regression)
        )

    def _infer_channels_by_hook(self):
        modules = list(self.yolo_model.model)
        target = modules[self.hook_module_idx]
        captured = {}

        def _hook(m, i, o):
            captured["feat"] = o

        h = target.register_forward_hook(_hook)
        try:
            dummy = torch.zeros(1, 3, self.input_size, self.input_size)
            _ = self.yolo_model(dummy)
        except Exception as e:
            print(f"⚠️ Erro durante infer_channels: {e}")
        finally:
            h.remove()

        feat = captured.get("feat", None)
        if feat is None:
            print("🔁 Usando fallback com 256 canais")
            return 256

        if isinstance(feat, (list, tuple)):
            for f in feat:
                if torch.is_tensor(f) and f.dim() == 4:
                    return f.shape[1]
        elif torch.is_tensor(feat):
            return feat.shape[1]
        return 256

    def forward(self, x, debug=False):
        modules = list(self.yolo_model.model)
        target = modules[self.hook_module_idx]
        captured = {}

        def _hook(m, i, o):
            captured["feat"] = o

        h = target.register_forward_hook(_hook)
        try:
            _ = self.yolo_model(x)
        finally:
            h.remove()

        feat = captured.get("feat", None)
        if feat is None:
            raise RuntimeError("❌ Falha ao capturar feature map. Verifique hook_module_idx.")

        if isinstance(feat, (list, tuple)):
            feat = next((f for f in feat if torch.is_tensor(f) and f.dim() == 4), None)
        if feat is None or not torch.is_tensor(feat):
            raise RuntimeError(f"Tipo inesperado de feature capturada: {type(feat)}")

        out = self.reg_head(feat)
        return out.squeeze(1) if out.shape[1] == 1 else out


# ============================================================
#  YOLO-Lite (versão super leve, para mobile)
# ============================================================
class YOLOLite_Det_Regression(nn.Module):
    """
    Versão leve inspirada no Tiny YOLO — otimizada para regressão.
    ~2.3M parâmetros, ideal para rodar em celular (CPU ou NPU).
    """
    def __init__(self, num_regression=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_regression)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.squeeze(1) if x.shape[1] == 1 else x


# ============================================================
#  Engine com paralelismo de CPU e DataLoader otimizado
# ============================================================
class Engine(Dataset):
    def __init__(self, dataframe, SETTINGS):
        super().__init__()

        num_threads = os.cpu_count()
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        print(f"🧠 PyTorch configurado para usar {num_threads} threads")

        self.SETTINGS = SETTINGS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DATA = dataframe
        self.k_folds = SETTINGS.K_FOLD
        self._cache = OrderedDict()
        self._max_cache_size = SETTINGS.MAX_CACHE
        self.scaler = SETTINGS.SCALER
        self.model_loaded = False
        self.__kFoldSplit__()

    def __len__(self):
        return len(self.DATA)

    def __get_image__(self, idx):
        img_path = self.DATA.iloc[idx]["file"]
        img = Image.open(img_path).convert("RGB")
        transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
        return transform(img)

    def __apply_transform__(self, img_tensor):
        img = img_tensor.permute(1, 2, 0).numpy()
        aug = A.Compose([
            A.RandomBrightnessContrast(0.1, 0.2, p=0.5),
            A.RGBShift(5, 5, 5, p=0.3),
            A.ISONoise(color_shift=(0.005, 0.01), intensity=(0.05, 0.1), p=0.8),
            A.Perspective(scale=(0.01, 0.03), p=0.3)
        ])
        img_aug = aug(image=(img * 255).astype(np.uint8))["image"]
        img_aug = torch.from_numpy(img_aug.astype(np.float32) / 255.0).permute(2, 0, 1)
        return img_aug

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        img = self.__get_image__(idx)
        if getattr(self, "USE_TRANSFORM", True):
            img = self.__apply_transform__(img)

        y_val = self.DATA.iloc[idx]["deltaH_cm"]
        if self.SETTINGS.USE_SCALER:
            y_val = self.SETTINGS.SCALER.transform(np.array([[y_val]]))[0][0]
        y = torch.tensor(y_val, dtype=torch.float32)

        res = (img, y)
        self._cache[idx] = res
        if len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)
        return res

    def __kFoldSplit__(self):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        self.SPLITS = [(tr, val) for tr, val in kf.split(self.DATA)]

    def set_parameters(self, batch_size, learning_rate, epochs, patience, model_name,
                       accumulation_steps=1, use_transform=True, weight_decay=1e-4, use_schedule=True):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.USE_TRANSFORM = use_transform
        self.model_name = model_name
        self.weight_decay = weight_decay
        self.use_schedule = use_schedule

    def load_model_frozen(self, custom_model=None, use_yolo_lite=False):
        if custom_model:
            self.model = custom_model
        elif use_yolo_lite:
            print("🪶 Usando YOLO-Lite (versão leve para regressão)")
            self.model = YOLOLite_Det_Regression().to(self.device)
        else:
            self.model = YOLOv8_Det_Regression().to(self.device)

        # Congela o backbone se YOLOv8
        if hasattr(self.model, "yolo_model"):
            for p in self.model.yolo_model.parameters():
                p.requires_grad = False
        for p in self.model.parameters():
            if p.requires_grad:
                p.requires_grad = True

        try:
            self.model = torch.compile(self.model)
            print("🚀 Modelo compilado com torch.compile()")

        except Exception as e:
            print(f"⚠️ torch.compile() não disponível: {e}")

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6) if self.use_schedule else None
        print(f"🧊 Modelo pronto. Parâmetros treináveis: {sum(p.numel() for p in params):,}")

    def run(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        for fold_idx, (tr_idx, val_idx) in enumerate(self.SPLITS):
            print(f"\n=== Treinando Fold {fold_idx+1}/{self.k_folds} ===")
            self.load_model_frozen(use_yolo_lite=True)  # 👈 usa YOLO-Lite por padrão

            num_workers = max(1, os.cpu_count() - 1)
            train_loader = DataLoader(
                Subset(self, tr_idx), batch_size=self.batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=False, prefetch_factor=2
            )
            val_loader = DataLoader(
                Subset(self, val_idx), batch_size=self.batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=False
            )

            best_val, _ = self._run_epochs(train_loader, val_loader, fold_idx)
        

        #rint(f"\n✅ Treinamento concluído. Melhor MAE: {best_overall:.4f}")

    def _run_epochs(self, train_loader, val_loader, fold_idx):
        criterion = nn.L1Loss()
        use_amp = torch.cuda.is_available()
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=use_amp)
        best_val = float("inf")
        patience = 0

        for epoch in tqdm(range(self.epochs), desc=f"Fold {fold_idx+1}", leave=False):
            self.model.train()
            tr_loss = 0.0

            # Treinamento
            for imgs, ys in train_loader:
                imgs, ys = imgs.to(self.device), ys.to(self.device)
                self.optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    preds = self.model(imgs)
                    loss = criterion(preds.squeeze(), ys) / self.accumulation_steps

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                tr_loss += loss.item() * imgs.size(0)

            tr_loss /= len(train_loader.dataset)

            # Validação
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, ys in val_loader:
                    imgs, ys = imgs.to(self.device), ys.to(self.device)
                    with autocast(enabled=use_amp):
                        preds = self.model(imgs)
                        loss = criterion(preds.squeeze(), ys)
                    val_loss += loss.item() * imgs.size(0)
            val_loss /= len(val_loader.dataset)

            if self.scheduler:
                self.scheduler.step()

            print(f"📘 Epoch {epoch+1:03d} | Train: {tr_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                patience = 0

                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"{self.model_name}_f{fold_idx}.pth"))
                
            else:
                patience += 1
                if patience >= self.patience:
                    print("⏹️ Early Stopping")
                    break

        return best_val, {}
