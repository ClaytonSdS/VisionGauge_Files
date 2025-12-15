import cv2
import matplotlib.pyplot as plt

# Caminho da imagem
image_path = r"dataset\dataset_od\6k\test\142.5\image_resized_6425_png.rf.28adb9b1f03363f4dd0a093576aa0a95.jpg"

# 1. Leitura da imagem (em BGR)
img_raw = cv2.imread(image_path)

# Converte BGR para RGB para exibir corretamente no matplotlib
img_raw_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

# 2. Redimensiona para 85x85
img_85 = cv2.resize(img_raw, (95, 95), interpolation=cv2.INTER_AREA)
img_85_rgb = cv2.cvtColor(img_85, cv2.COLOR_BGR2RGB)

# 3. Redimensiona a imagem 85x85 para 120x120
img_120 = cv2.resize(img_85, (120, 120), interpolation=cv2.INTER_AREA)
img_120_rgb = cv2.cvtColor(img_120, cv2.COLOR_BGR2RGB)

# 4. Plotando os subplots
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_raw_rgb)
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_85_rgb)
plt.title("Resize 85x85")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_120_rgb)
plt.title("Resize 85x85 -> 120x120")
plt.axis("off")

plt.tight_layout()
plt.show()
