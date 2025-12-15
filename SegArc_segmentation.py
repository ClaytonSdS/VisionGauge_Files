import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Carrega o modelo
model_path = "models/SegARC/weights/best.pt"
model = YOLO(model_path)

# Pastas
image_dir = "dataset/processed"
image_dir = "dataset/raw"

save_dir = "segmentation"
os.makedirs(save_dir, exist_ok=True)

# Loop das imagens
for image_name in os.listdir(image_dir):

    if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(image_dir, image_name)

    # Carrega imagem uma vez
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Predição
    results = model.predict(img_rgb, conf=0.2)

    # Imagem anotada
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Salva imagem com bounding boxes
    #cv2.imwrite(os.path.join(save_dir,f"{os.path.splitext(image_name)[0]}_imagem_completa_boxes.png"), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

    # Extrair boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    h, w = img_rgb.shape[:2]

    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)

        # Corrige limites
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        crop = img_rgb[y1:y2, x1:x2]

        filename = f"{os.path.splitext(image_name)[0]}_box_{i+1}_class_{int(cls)}_conf_{score:.2f}.png"
        cv2.imwrite(os.path.join(save_dir, filename),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))