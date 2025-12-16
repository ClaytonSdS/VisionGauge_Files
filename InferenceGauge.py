from Gauge import VisionGauge
import os
import pandas as pd
from Arch import NewDirectModel_Inference as NDM
from ultralytics import YOLO

# Caminhos dos modelos
segmentation_model_path = "models/SegARC_v04_lr0.0001_5k/weights/best.pt"
regressor_model_path = r"models\RegArc\EfficientNet_678\efficientnet_lite_120x120.pth"

# Lendo os caminhos das imagens do dataset
dataframe = pd.read_csv(r"dataset/testing/dataset_testing_paths.csv")
paths = dataframe['file'].tolist()

# Instanciando a classe VisionGauge e passando o dataframe
vision_gauge = VisionGauge(
    segmentation_model_path=segmentation_model_path,
    regressor_model_path=regressor_model_path,
    batch_size=16,  
    image_size=(120, 120), 
    dataset_path="dataset/testing/",  # Caminho onde estão as imagens
    dataframe=dataframe  # Passando o dataframe com os caminhos e as labels
)

# Realizando a predição
df_predictions = vision_gauge.predict_paths(image_paths=paths)

# Exibindo as previsões
print(df_predictions)

# Salvando as predições em um CSV
output_dir = r"dataset/testing"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "predictions.csv")
df_predictions.to_csv(save_path, index=False)
