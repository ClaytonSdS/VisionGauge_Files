import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# =========================
# CONFIGURAÇÕES
# =========================
MODEL_PATHS = {
    "YOLOv8s": "models\\Yolo_v8\\results.csv",
    "YOLO11s": "models\\Yolov11s\\results.csv",
    "YOLO26s": "models\\Yolo_v26s\\results.csv",
}

LOSS_CURVES = [
    ("train/box_loss", "Perda de Caixa (Treino)", "Perda de Caixa"),
    ("train/cls_loss", "Perda de Classe (Treino)", "Perda de Classe"),
    ("train/dfl_loss", "Perda DFL (Treino)", "Perda DFL"),
    ("val/box_loss", "Perda de Caixa (Validação)", "Perda de Caixa"),
    ("val/cls_loss", "Perda de Classe (Validação)", "Perda de Classe"),
    ("val/dfl_loss", "Perda DFL (Validação)", "Perda DFL"),
]

MAP_METRICS = [
    ("metrics/mAP50(B)", "mAP@50"),
    ("metrics/mAP50-95(B)", "mAP@50-95"),
]

def load_results(paths):
    models = {}
    for name, path in paths.items():
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        models[name] = df
    return models

models = load_results(MODEL_PATHS)

cmap = cm.get_cmap("tab10")
colors = {name: cmap(i) for i, name in enumerate(models.keys())}
fig2, axs2 = plt.subplots(1, 2, figsize=(10, 4))

for i, (ax, (col, title)) in enumerate(zip(axs2, MAP_METRICS)):
    for name, df in models.items():
        ax.plot(df["epoch"], df[col], linewidth=2, label=name, color=colors[name])

    ax.set_title(f"{chr(97 + i)}) {title}", loc="left", fontsize=14)
    ax.set_xlabel("Época", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)

    ax.tick_params(axis='both', labelsize=10)
    ax.grid(linestyle=':', alpha=0.4)

handles, labels = axs2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, frameon=False)

fig2.tight_layout(rect=[0, 0, 1, 0.9])

fig2.savefig(
    "models\\detector_map_comparison_1x2.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig2)