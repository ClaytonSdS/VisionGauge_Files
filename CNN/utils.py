import torch
import numpy as np
from mlflow.models.signature import ModelSignature, infer_signature
from tqdm.auto import tqdm
import albumentations as A
from PIL import Image

def print_logs(isthebest: bool = False, **kwargs):
    logs = kwargs.get("logs", {})

    if len(logs) > 0:
        for key, value in logs.items():
            print(f"{key}: {value:.4f}", end=" | ")

        if isthebest:
            print("(*)", end="")

        print()
    

def inferir_assinatura(loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, ModelSignature]:
    batch = next(iter(loader))
    if isinstance(batch, (list, tuple)):
        img_sample, y_sample = batch[:2]
    else:
        raise ValueError("Batch inesperado")

    X = img_sample.cpu().numpy().astype("float32")
    y = y_sample.cpu().numpy()

    signature = infer_signature(X, y)

    return X, signature

# Gerar augmented images e retornar o dict para controle e governança
def GEN_AUGMENTED_IMAGES(image_path:str, albumentations: A.Compose) -> np.array:
    # Abrir imagem
    image = np.array(Image.open(image_path))

    # Aplicar transformações
    transformed = albumentations(image=image)["image"]

    albumentations_dict = A.to_dict(albumentations)
    
    # Retornar a imagem
    return transformed, albumentations_dict
    


def train_one_epoch(model, train_loader:torch.utils.data.DataLoader, criterion, optimizer, epoch:int, device:torch.device, scaler_amp=None) -> float:
        model.train()
        train_loss = 0.0
    
        for img, y in tqdm(train_loader, desc=f"Train", leave=False):
            img = img.to(device, non_blocking=True).float()
            #print("IMG DEVICE:", img.device)
            y = y.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
    
            outputs = model(img).squeeze(-1)
            loss = criterion(outputs, y)
    
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item() * img.size(0)
    
        avg_loss = train_loss / len(train_loader.dataset)
    
        return avg_loss


def validate_one_epoch(model, val_loader:torch.utils.data.DataLoader, criterion, epoch:int, device:torch.device) -> float:
        model.eval()
        val_loss = 0.0
    
        with torch.no_grad():
            for img, y in tqdm(val_loader, desc="Valid", leave=False):
                img = img.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).float()
    
                outputs = model(img).squeeze(-1)
                loss = criterion(outputs, y)
    
                val_loss += loss.item() * img.size(0)
    
        return val_loss / len(val_loader.dataset)

def test_one_epoch(model, test_loader:torch.utils.data.DataLoader, criterion, epoch:int, device:torch.device) -> float:
        model.eval()
        test_loss = 0.0
    
        with torch.no_grad():
            for img, y in tqdm(test_loader, desc="Test", leave=False):
                img = img.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True).float()
    
                outputs = model(img).squeeze(-1)
                loss = criterion(outputs, y)
    
                test_loss += loss.item() * img.size(0)
    
        return test_loss / len(test_loader.dataset)