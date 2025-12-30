import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm

folder_path = "dataset/training/cropped"
save_path = "dataset/training/resized"

my_current_dir = os.getcwd()
folder_path = os.path.join(my_current_dir, folder_path)
save_path = os.path.join(my_current_dir, save_path)
USE_CLAHE = False

target_shape = (1000, 1000)

# cria o objeto CLAHE
if USE_CLAHE:   
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for filename in tqdm.tqdm(os.listdir(folder_path)):
    img_num = filename.split("_")[-1].split(".")[0]

    img = cv2.imread(os.path.join(folder_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # converte para LAB
    if USE_CLAHE:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        L_eq = clahe.apply(L)
        lab_eq = cv2.merge((L_eq, A, B))
        img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2RGB)
        img_resized = cv2.resize(img_eq, (target_shape[1], target_shape[0]))

    else:
        img_resized = cv2.resize(img, (target_shape[1], target_shape[0]))   


    # salvar imagem
    plt.imsave(os.path.join(save_path, f"image_resized_{img_num}.png"), img_resized)

print("Finalizado o redimensionamento das imagens.")
