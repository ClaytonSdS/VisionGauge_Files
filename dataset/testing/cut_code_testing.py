import os
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import tqdm 
import os
import re

folder_path = "dataset//testing//hopper"
save_path = "dataset//testing//processed"
initial_number = 482

my_current_dir = os.getcwd()
folder_path = os.path.join(my_current_dir, folder_path)
save_path = os.path.join(my_current_dir, save_path)


target_shape = (2000, 2000) # altura, largura

def crop_bottom_center(
    img,
    crop_h=1500,
    crop_w=1500,
    horizontal_shift=0,   # negativo = esquerda, positivo = direita
    start_row=None        # None = começa no fundo (default)
):
    height, width, _ = img.shape

    # centro horizontal da imagem
    cx = width // 2

    # aplica o deslocamento horizontal
    cx = cx + horizontal_shift

    # limita para não sair da imagem
    cx = max(crop_w // 2, min(width - crop_w // 2, cx))

    # linha inicial (se None, usa o fundo padrão)
    if start_row is None:
        start_row = height - crop_h

    # garante que não passe dos limites
    start_row = max(0, min(height - crop_h, start_row))


    end_row = start_row + crop_h

    start_col = cx - crop_w // 2
    end_col = cx + crop_w // 2

    return img[start_row:end_row, start_col:end_col]


count = 1

def adjust_temperature(img, i=0):
    # img vem em RGB — converter para BGR para manipular
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    b, g, r = cv2.split(bgr)

    # intensidade > 0 = esfria
    # intensidade < 0 = aquece
    b = cv2.add(b,  i)      # mais azul
    r = cv2.subtract(r, i)  # menos vermelho

    out = cv2.merge((b, g, r))
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    
    return out

import random
for filename in tqdm.tqdm(os.listdir(folder_path)):

    img = cv2.imread(os.path.join(folder_path, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img= cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img= cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img= cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    delta_h = random.randint(-100, 10)
    delta_x = random.randint(-20, 20)
    delta_crop = random.randint(-200, 700)

    if count <= 9004:
        recorte = crop_bottom_center(
            adjust_temperature(img, i=0), # i > 0 -> -Temp | i < 0 -> +Temp
            crop_h=1100 + delta_crop,
            crop_w=1100 + delta_crop,
            start_row=1100 + delta_h,
            horizontal_shift= -400+ delta_x
        )
        # resize para 120 120
        

    #recorte = cv2.resize(img, (120, 120))
    plt.imsave(os.path.join(save_path, f"image_test_{initial_number}.png"), recorte)

    count += 1
    initial_number += 1

print("Finalizado o recorte das imagens.")