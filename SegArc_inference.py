import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import numpy as np

# 1) Caminhos dos modelos
model_path_1 = r"models/SegARC_v00/weights/best.pt"   
model_path_2 = r"models\SegARC_v01\weights\best.pt"
model_path_3 = r"models\SegARC_v02\weights\best.pt"


model1 = YOLO(model_path_1)
model2 = YOLO(model_path_2)
model3 = YOLO(model_path_3)  

# Imagem com sombra
image_path = "dataset/dataset_od/UTM_Dataset_OD.v5-3k.yolov8/test/images/image_resized_1667_png.rf.a528edf6336756c922f94588362e27c9.jpg"
image_path = r"dataset\dataset_od\UTM_Dataset_OD.v5-3k.yolov8\train\images\image_resized_1910_png.rf.4443d62d2bb4862ebbb3bb8508ca9b39.jpg"

# 3) Execução dos 3 modelos
results1 = model1.predict(image_path, conf=0.2)
results2 = model2.predict(image_path, conf=0.2)
results3 = model3.predict(image_path, conf=0.5)   

# 4) Cria pasta para salvar as imagens
save_dir = "outputs_3_models"
os.makedirs(save_dir, exist_ok=True)

# 5) Carrega imagem original (RGB)
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 6) Gera imagens anotadas
annot1_bgr = results1[0].plot()
annot2_bgr = results2[0].plot()
annot3_bgr = results3[0].plot()  

annot1_rgb = cv2.cvtColor(annot1_bgr, cv2.COLOR_BGR2RGB)
annot2_rgb = cv2.cvtColor(annot2_bgr, cv2.COLOR_BGR2RGB)
annot3_rgb = cv2.cvtColor(annot3_bgr, cv2.COLOR_BGR2RGB)

# Extrai boxes de cada modelo
boxes1 = results1[0].boxes.xyxy.cpu().numpy()
classes1 = results1[0].boxes.cls.cpu().numpy()
scores1 = results1[0].boxes.conf.cpu().numpy()

boxes2 = results2[0].boxes.xyxy.cpu().numpy()
classes2 = results2[0].boxes.cls.cpu().numpy()
scores2 = results2[0].boxes.conf.cpu().numpy()

boxes3 = results3[0].boxes.xyxy.cpu().numpy()  
classes3 = results3[0].boxes.cls.cpu().numpy()
scores3 = results3[0].boxes.conf.cpu().numpy()

# Salva imagens anotadas
def save_crops(save_folder, img, boxes, classes, scores, model_name):
    os.makedirs(save_folder, exist_ok=True)

    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = box.astype(int)

        # Recorta o bounding box
        crop = img[y1:y2, x1:x2]

        # Nome do arquivo
        filename = f"{model_name}_box{i}_cls{int(cls)}_conf{score:.2f}.jpg"
        path = os.path.join(save_folder, filename)

        cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

save_crops(os.path.join(save_dir, "model1_crops"), img_rgb, boxes1, classes1, scores1, "model1")
save_crops(os.path.join(save_dir, "model2_crops"), img_rgb, boxes2, classes2, scores2, "model2")
save_crops(os.path.join(save_dir, "model3_crops"), img_rgb, boxes3, classes3, scores3, "model3")

print("Imagens anotadas salvas em:", save_dir)



fig = plt.figure(figsize=(21, 7))  

# Modelo 1
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(annot1_rgb)
ax1.set_title("Modelo 1 – v00")
ax1.axis("off")

# Modelo 2
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(annot2_rgb)
ax2.set_title("Modelo 2 – 1k_augmented")
ax2.axis("off")

# Modelo 3
ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(annot3_rgb)
ax3.set_title("Modelo 3 – 2k_augmented")  
ax3.axis("off")

plt.tight_layout()
plt.show()
