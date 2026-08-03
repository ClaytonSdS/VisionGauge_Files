
import pandas as pd

yolov8s_results = pd.read_csv("models\\Yolo_v8\\results.csv")
yolo11s_results = pd.read_csv("models\\Yolo_v11s\\results.csv")
yolo26s_results = pd.read_csv("models\\Yolo_v26s\\results.csv")

# DATAFRAMES COM AS COLUNAS:
# epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2


# Gerar duas figuras um plot com 2x2 com train/box_loss x epoch, train/cls_loss x epoch, train/dfl_loss x epoch, e val/box_loss x epoch


# Gerar uma figura (1x2) com 3 curvas de mAP50(B) x epoch para os 3 modelos e metrics/mAP50-95(B) x epoch para os 3 modelos