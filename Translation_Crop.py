import numpy as np
import cv2
import os
import tqdm

image_path = "dataset\\training\\resized\\image_resized_448.png"
my_current_dir = os.getcwd()
set = "valid"
fold = "3k_augmented"

zooms = [0.9, 1.2, 1.4, 1.6]

global count

# Função para aplicar zoom:
def apply_zoom(img, zoom_factor):
    h, w, _ = img.shape
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)

    img_resized = cv2.resize(img, (new_w, new_h))

    # Crop ou pad para voltar ao tamanho original
    if zoom_factor < 1.0:
        # Pad
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        padded = np.zeros((h, w, 3), dtype=img.dtype)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        return padded
    
    else:
        # Crop
        start_row = (new_h - h) // 2
        start_col = (new_w - w) // 2
        return img_resized[start_row:start_row + h, start_col:start_col + w]


# Percorrer todas as imagens na pasta 3k/train
for folder in tqdm.tqdm(os.listdir(os.path.join(my_current_dir, fold, set))):
    current_folder = os.path.join(my_current_dir, fold, set, folder)

    for image in os.listdir(current_folder):
        if image.split("_")[0].startswith("image"):
            image_path = os.path.join(current_folder, image)
            image_reader = cv2.imread(image_path)
            image_number = image.split("_")[2]

            # Aplicar zooms
            for zoom in zooms:
                img_zoomed = apply_zoom(image_reader, zoom)

                # resize
                img_zoomed = cv2.resize(img_zoomed, (224, 224))

                # salvar a imagem
                zoom_str = str(zoom).replace(".", "_")
                image_name = "zoom_" + zoom_str + "image" + image_number
                save_path = os.path.join(my_current_dir, fold, set, folder,f"{image_name}.jpg")

                # salvar imagem
                cv2.imwrite(save_path, img_zoomed)

# Função para cortar a image  e pular linhas
def crop_skip_lines(img, skip=100, image_size=400):
    h, w, _ = img.shape

    cropped = img[skip:skip + image_size, :image_size, :]


    return cropped


target = 500 - 400 # -- 276
skips = [i for i in range(0, target, 5)]

# plotar a primeira imagem com os cortes
count = 0
for folder in tqdm.tqdm(os.listdir(os.path.join(my_current_dir, fold, set))):
    current_folder = os.path.join(my_current_dir, fold, set, folder)

    for image in os.listdir(current_folder):
        if image.split("_")[0].startswith("image"):
            image_path = os.path.join(current_folder, image)
            image_reader = cv2.imread(image_path)
            image_number = image.split("_")[2]

            # Aplicar cortes pulando linhas
            for skip in skips:
                img_cropped = crop_skip_lines(image_reader, skip=skip)

                # resize
                img_cropped = cv2.resize(img_cropped, (224, 224))

                # salvar a imagem
                skip_str = str(skip)
                image_name = "skip_" + skip_str + "_image" + image_number
                save_path = os.path.join(my_current_dir, fold, set, folder,f"{image_name}.jpg")
                # salvar imagem
                cv2.imwrite(save_path, img_cropped)
            
print("Finalizado o aumento de dados com zoom.")    