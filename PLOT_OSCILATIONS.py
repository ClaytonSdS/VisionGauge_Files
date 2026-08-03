import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

models = ['ResNet-18', 'EfficientNet-B0',
       'MobileNetV3 Small', 'MobileNetV3 Large']

def calculate_deltah(dataframe, tau:float=0.2):
    true_height = dataframe['true_height']
    resnet18 = dataframe['ResNet-18']
    efficient = dataframe['EfficientNet-B0']
    mobv3s = dataframe['MobileNetV3 Small']
    mobv3l = dataframe['MobileNetV3 Large']

    # delta_h
    dataframe['deltaH_ResNet18'] = abs(resnet18 - true_height)
    dataframe['deltaH_EfficientNetB0'] = abs(efficient - true_height)
    dataframe['deltaH_MobileNetV3Small'] = abs(mobv3s - true_height)
    dataframe['deltaH_MobileNetV3Large'] = abs(mobv3l - true_height)

    # phi_f
    dataframe['phi_f_ResNet18'] = (np.abs(dataframe['deltaH_ResNet18'] ) <= tau).astype(int)
    dataframe['phi_f_EfficientNetB0'] = (np.abs(dataframe['deltaH_EfficientNetB0'] ) <= tau).astype(int)
    dataframe['phi_f_MobileNetV3Small'] = (np.abs(dataframe['deltaH_MobileNetV3Small'] ) <= tau).astype(int)
    dataframe['phi_f_MobileNetV3Large'] = (np.abs(dataframe['deltaH_MobileNetV3Large'] ) <= tau).astype(int)

    # PHI
    dataframe['PHI_ResNet18'] = dataframe['phi_f_ResNet18'].mean()
    dataframe['PHI_EfficientNetB0'] = dataframe['phi_f_EfficientNetB0'].mean()
    dataframe['PHI_MobileNetV3Small'] = dataframe['phi_f_MobileNetV3Small'].mean()
    dataframe['PHI_MobileNetV3Large'] = dataframe['phi_f_MobileNetV3Large'].mean()


    return dataframe

import pandas as pd

detector_names = ['yolov8s', 'yolo11s', 'yolo26s']

for detector_name in detector_names:
    Dataframes = {
        20: {
            20: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F20_all_detectors.csv")),

            40: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F40_all_detectors.csv")),

            60: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F60_all_detectors.csv")),

            80: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F80_all_detectors.csv")),

            100: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F100_all_detectors.csv")),

            120: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F120_all_detectors.csv")),

            140: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F140_all_detectors.csv"))
        },

        25: {20: [], 40: [], 60: [], 80: [], 100: [], 120: [], 140: []},

        30: {
            20: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F20_all_detectors.csv")),

            40: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F40_all_detectors.csv")),

            60: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F60_all_detectors.csv")),

            80: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F80_all_detectors.csv")),

            100: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F100_all_detectors.csv")),

            120: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F120_all_detectors.csv")),

            140: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F140_all_detectors.csv"))
        },

        35: {20: [], 40: [], 60: [], 80: [], 100: [], 120: [], 140: []},

        40: {
            20: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F20_all_detectors.csv")),

            40: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F40_all_detectors.csv")),

            60: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F60_all_detectors.csv")),

            80: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F80_all_detectors.csv")),

            100: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F100_all_detectors.csv")),

            120: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F120_all_detectors.csv")),

            140: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F140_all_detectors.csv"))
        }
    }

    for interpolate in [25, 35]:
        for fita in [20, 40, 60, 80, 100, 120, 140]:
            df1 = Dataframes[interpolate - 5][fita]
            df2 = Dataframes[interpolate + 5][fita]
            df2 = df2.reindex_like(df1)
            df_interp = (df1 + df2) / 2
            Dataframes[interpolate][fita] = df_interp

    for distancia in list(Dataframes.keys()):
        for fita in [20, 40, 60, 80, 100, 120, 140]:
            Dataframes[distancia][fita] = calculate_deltah(Dataframes[distancia][fita], tau=0.2)

    # PLOTAR GRAFICO COMPLETO
    window_size = 20
    ht_values = [20, 40, 60, 80, 100, 120, 140]
    tapes = list(Dataframes.keys())
    reg_models = ['ResNet-18', 'EfficientNet-B0', 'MobileNetV3 Small', 'MobileNetV3 Large']

    lines = [(0, (3, 10, 1, 10, 1, 10)), 'solid', 'dashed', 'dashdot', 'dotted']
    markers = ['o', 's', '^', 'x', '*', 'D', '+']
    colors = [
        "#FF7F7F",
        "#7FBFFF",
        "black",
        "#b7cc18",
        "#012677"
    ]

    fig, axes = plt.subplots(nrows=len(ht_values), ncols=len(reg_models), figsize=(15, 20), sharex=True)

    if len(ht_values) == 1:
        axes = [axes]
    if len(reg_models) == 1:
        axes = [[ax] for ax in axes]

    for i, ht in enumerate(ht_values):
        for j, model in enumerate(reg_models):
            ax = axes[i][j]
            for idx, tape in enumerate(tapes):
                ax.plot(
                    Dataframes[tape][ht][model].rolling(window=window_size).mean(),
                    label=f'$d$ = {tape} cm',
                    linestyle=lines[idx % len(lines)],
                    marker=markers[idx % len(markers)], linewidth=1,
                    markevery=15, markersize=5,
                    color=colors[idx % len(colors)]
                )

            if i == 0:
                ax.set_title(f"{chr(97+i)}) {model}", loc='left', fontsize=20)
            if j == 0:
                ax.set_ylabel(f'$h_t = {ht}$ cm', fontsize=14)
            if i == len(ht_values)-1:
                ax.set_xlabel('Frame ($f$)', fontsize=14)
            y_formatter = ScalarFormatter(useOffset=False)
            y_formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.grid(alpha=0.3, linestyle=':')

    legend_handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc='upper center',
        ncol=len(tapes),
        fontsize=14,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"dataset\\testing\\Oscilation\\oscilation_raw_{detector_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: oscilation_raw_{detector_name}.png")

#=========================
# PLOT 2
print("Plotando gráfico de oscilação 2")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fitas = [20, 40, 60, 80, 100, 120, 140]
taus = [0.15, 0.20, 0.25]
heatmap_distances = [20, 30, 40]

def build_plot2_dataframes(detector_name):
    Dataframes = {
        20: {
            20: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F20_all_detectors.csv")),
            40: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F40_all_detectors.csv")),
            60: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F60_all_detectors.csv")),
            80: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F80_all_detectors.csv")),
            100: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F100_all_detectors.csv")),
            120: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F120_all_detectors.csv")),
            140: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D20_F140_all_detectors.csv"))
        },
        25: {20: [], 40: [], 60: [], 80: [], 100: [], 120: [], 140: []},
        30: {
            20: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F20_all_detectors.csv")),
            40: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F40_all_detectors.csv")),
            60: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F60_all_detectors.csv")),
            80: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F80_all_detectors.csv")),
            100: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F100_all_detectors.csv")),
            120: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F120_all_detectors.csv")),
            140: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D30_F140_all_detectors.csv"))
        },
        35: {20: [], 40: [], 60: [], 80: [], 100: [], 120: [], 140: []},
        40: {
            20: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F20_all_detectors.csv")),
            40: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F40_all_detectors.csv")),
            60: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F60_all_detectors.csv")),
            80: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F80_all_detectors.csv")),
            100: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F100_all_detectors.csv")),
            120: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F120_all_detectors.csv")),
            140: (lambda df: df[df['detector'] == detector_name].drop(columns=['detector']).reset_index(drop=True))
                 (pd.read_csv("dataset\\testing\\Oscilation\\Oscilation_D40_F140_all_detectors.csv"))
        }
    }

    for interpolate in [25, 35]:
        for fita in fitas:
            df1 = Dataframes[interpolate - 5][fita]
            df2 = Dataframes[interpolate + 5][fita]
            df2 = df2.reindex_like(df1)
            Dataframes[interpolate][fita] = (df1 + df2) / 2

    return Dataframes

# =========================
# FUNÇÃO (igual a sua)
# =========================
def calculate_deltah(dataframe, tau: float):
    dataframe = dataframe.copy()

    true_height = dataframe['true_height']

    dataframe['deltaH_ResNet18'] = abs(dataframe['ResNet-18'] - true_height)
    dataframe['deltaH_EfficientNetB0'] = abs(dataframe['EfficientNet-B0'] - true_height)
    dataframe['deltaH_MobileNetV3Small'] = abs(dataframe['MobileNetV3 Small'] - true_height)
    dataframe['deltaH_MobileNetV3Large'] = abs(dataframe['MobileNetV3 Large'] - true_height)

    dataframe['phi_f_ResNet18'] = (dataframe['deltaH_ResNet18'] <= tau).astype(int)
    dataframe['phi_f_EfficientNetB0'] = (dataframe['deltaH_EfficientNetB0'] <= tau).astype(int)
    dataframe['phi_f_MobileNetV3Small'] = (dataframe['deltaH_MobileNetV3Small'] <= tau).astype(int)
    dataframe['phi_f_MobileNetV3Large'] = (dataframe['deltaH_MobileNetV3Large'] <= tau).astype(int)

    dataframe['PHI_ResNet18'] = dataframe['phi_f_ResNet18'].mean()
    dataframe['PHI_EfficientNetB0'] = dataframe['phi_f_EfficientNetB0'].mean()
    dataframe['PHI_MobileNetV3Small'] = dataframe['phi_f_MobileNetV3Small'].mean()
    dataframe['PHI_MobileNetV3Large'] = dataframe['phi_f_MobileNetV3Large'].mean()

    return dataframe


# =========================
# INTERPOLAÇÃO (25 e 35)
# =========================
# for interpolate in [25, 35]:
#     for fita in fitas:
#         df1 = Dataframes[interpolate - 5][fita]
#         df2 = Dataframes[interpolate + 5][fita]
#         df2 = df2.reindex_like(df1)
#         Dataframes[interpolate][fita] = (df1 + df2) / 2


# =========================
# FIGURA COM 3 SUBPLOTS
# =========================
modelos = {
    "ResNet-18": "PHI_ResNet18",
    "EfficientNet-B0": "PHI_EfficientNetB0",
    "MobileNetV3 Small": "PHI_MobileNetV3Small",
    "MobileNetV3 Large": "PHI_MobileNetV3Large"
}

taus = [0.15, 0.20, 0.25]
fitas = [20, 40, 60, 80, 100, 120, 140]

FONTS_SIZE = {
    "title": 25,
    "label": 18,
    "ticks": 18,
    "legend": 18,
    "annotation_text": 18,
    "marker_text": 18,
    "text": 19,
}

for detector_name in detector_names:
    Dataframes = build_plot2_dataframes(detector_name)

    fig, axes = plt.subplots(len(modelos), len(taus), figsize=(20, 26), sharey=True)
    fig.subplots_adjust(right=0.88)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgray')

    count = 0
    for row, (modelo_nome, phi_col) in enumerate(modelos.items()):
        for col, tau in enumerate(taus):
            try:
                ax = axes[row, col]

                Data_proc = {}
                for dist in heatmap_distances:
                    Data_proc[dist] = {}
                    for fita in fitas:
                        Data_proc[dist][fita] = calculate_deltah(Dataframes[dist][fita], tau)

                X_vals = heatmap_distances
                Y_vals = fitas
                Z = np.full((len(Y_vals), len(X_vals)), np.nan)

                for j, x in enumerate(X_vals):
                    for i, y in enumerate(Y_vals):
                        if not Data_proc[x][y].empty:
                            Z[i, j] = Data_proc[x][y][phi_col].iloc[0]

                im = ax.imshow(Z, cmap=cmap, vmin=0, vmax=1, aspect='auto')

                ax.set_xticks(range(len(X_vals)), X_vals)
                ax.tick_params(axis='x', labelsize=FONTS_SIZE['text'])

                ax.set_yticks(range(len(Y_vals)), labels=Y_vals)
                ax.tick_params(axis='y', labelsize=FONTS_SIZE['text'])

                if row == len(modelos) - 1:
                    ax.set_xlabel(r"$d$ (cm)", fontsize=FONTS_SIZE['text'])
                if col == 0:
                    ax.set_ylabel("Fluid Height (cm)", fontsize=FONTS_SIZE['text'])

                ax.set_title(rf"{chr(97+count)}) {modelo_nome} ($\tau = {tau}$)", loc='left', fontsize=FONTS_SIZE['title'])
                count += 1

                mean_val = np.nanmean(Z)*100
                print(f"{detector_name} | tau = {tau} | {modelo_nome}| {mean_val:.2f}")

                for i in range(len(Y_vals)):
                    for j in range(len(X_vals)):
                        if np.isnan(Z[i, j]):
                            ax.text(j, i, "--",
                                    ha="center", va="center", color="black", fontsize=FONTS_SIZE['text'])
                        else:
                            color = "black" if Z[i, j] > 0.6 else "white"
                            ax.text(j, i, f"{Z[i, j]:.2f}",
                                    ha="center", va="center", color=color, fontsize=FONTS_SIZE['text'])
            except Exception as e:
                print(f"[ERROR] Failed processing {detector_name} | tau = {tau} | {modelo_nome}: {str(e)}")
                import traceback
                traceback.print_exc()

            for i in range(len(Y_vals)):
                for j in range(len(X_vals)):
                    if np.isnan(Z[i, j]):
                        ax.text(j, i, "--",
                                ha="center", va="center", color="black", fontsize=FONTS_SIZE['text'])
                    else:
                        color = "black" if Z[i, j] > 0.6 else "white"
                        ax.text(j, i, f"{Z[i, j]:.2f}",
                                ha="center", va="center", color=color, fontsize=FONTS_SIZE['text'])

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\Phi$", fontsize=FONTS_SIZE['text']+10)
    cbar.ax.tick_params(labelsize=FONTS_SIZE['text'])

    fig.subplots_adjust(
        left=0.06,
        right=0.88,
        top=0.92,
        bottom=0.07,
        wspace=0.12,
        hspace=0.35
    )

    plt.savefig(f"dataset\\testing\\Oscilation\\oscilation_results_{detector_name}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo: oscilation_results_{detector_name}.png")
