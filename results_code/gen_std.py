import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DISTANCE_20 = [
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D20_F20_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D20_F40_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D20_F60_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D20_F80_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D20_F100_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D20_F120_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D20_F140_all_detectors.csv",
]

DISTANCE_30 = [
    r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D30_F20_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D30_F40_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D30_F60_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D30_F80_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D30_F100_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D30_F120_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D30_F140_all_detectors.csv",
]

DISTANCE_40 = [
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D40_F20_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D40_F40_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D40_F60_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D40_F80_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D40_F100_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D40_F120_all_detectors.csv",
	r"C:\Users\Clayton\Desktop\VisionGauge_Files\dataset\testing\Oscilation\Oscilation_D40_F140_all_detectors.csv",
]

# Juntar tudo em um unico df
def load_and_concat_csv(file_paths):
	dfs = []
	for path in file_paths:
		df = pd.read_csv(path)
		file_name = os.path.basename(path)
		match_distance = re.search(r"_D(\d+)_", file_name)
		df['distance'] = int(match_distance.group(1)) if match_distance else np.nan
		dfs.append(df)
	return pd.concat(dfs, ignore_index=True)

df_distance_20 = load_and_concat_csv(DISTANCE_20)
df_distance_30 = load_and_concat_csv(DISTANCE_30)
df_distance_40 = load_and_concat_csv(DISTANCE_40)

# Juntar tudo em um unico df
df_all = pd.concat([df_distance_20, df_distance_30, df_distance_40], ignore_index=True)

backbones = ['ResNet-18', 'EfficientNet-B0', 'MobileNetV3 Small', 'MobileNetV3 Large']
detectors = ['yolov8s', 'yolo11s', 'yolo26s']
fluid_heights = [20, 40, 60, 80, 100, 120, 140]
distances = [20, 30, 40]
TAU = 0.25
TAUS = [0.15, 0.20, 0.25]
OUTPUT_DIR = 'metrics_txt'
FONTS_SIZE = {
	'title': 24,
	'text': 22,
	'cell': 23,
	'cbar_label': 26,
	'cbar_ticks': 23,
}


def compute_oscillation_metrics(df, pred_column, true_column='true_height', tau=TAU):
	"""Calcula Phi (score global) e sigma conforme as equacoes do texto."""
	valid = df[[pred_column, true_column]].dropna()
	if valid.empty:
		return np.nan, np.nan, 0

	delta_h = valid[pred_column] - valid[true_column]
	phi_frames = (delta_h.abs() <= tau).astype(float)
	phi_global = phi_frames.mean()
	sigma = np.sqrt(((phi_frames - phi_global) ** 2).mean())
	return phi_global, sigma, len(phi_frames)


def model_report_filename(model_name):
	return f"metrics_{model_name.replace(' ', '_')}.txt"


def build_phi_matrix(df_detector, backbone, tau):
	"""Monta matriz de Phi com eixo y=fluid_heights e eixo x=distances."""
	Z = np.full((len(fluid_heights), len(distances)), np.nan)
	for j, dist in enumerate(distances):
		for i, height in enumerate(fluid_heights):
			df_pair = df_detector[
				(df_detector['distance'] == dist) &
				(df_detector['true_height'] == height)
			]
			phi_val, _, _ = compute_oscillation_metrics(
				df_pair,
				pred_column=backbone,
				true_column='true_height',
				tau=tau,
			)
			Z[i, j] = phi_val
	return Z


def plot_detector_heatmaps(detector_name, df_detector, output_dir):
	"""Plota grade 4x3 de heatmaps (4 backbones x 3 taus) para um detector."""
	fig, axes = plt.subplots(len(backbones), len(TAUS), figsize=(20, 26), sharey=True)
	fig.subplots_adjust(right=0.88)

	count = 0
	im = None
	for row, backbone in enumerate(backbones):
		for col, tau in enumerate(TAUS):
			ax = axes[row, col]
			Z = build_phi_matrix(df_detector, backbone, tau)

			im = ax.imshow(Z, cmap='plasma', vmin=0, vmax=1, aspect='auto')

			ax.set_xticks(range(len(distances)), distances)
			ax.tick_params(axis='x', labelsize=FONTS_SIZE['text'])

			ax.set_yticks(range(len(fluid_heights)), labels=fluid_heights)
			ax.tick_params(axis='y', labelsize=FONTS_SIZE['text'])

			if row == len(backbones) - 1:
				ax.set_xlabel(r"$d$ (cm)", fontsize=FONTS_SIZE['text'])
			if col == 0:
				ax.set_ylabel("Altura Manométrica (cm)", fontsize=FONTS_SIZE['text'])

			ax.set_title(
				rf"{chr(97 + count)}) {backbone} ($\tau = {tau}$)",
				loc='left',
				fontsize=FONTS_SIZE['title']
			)
			count += 1

			for i in range(len(fluid_heights)):
				for j in range(len(distances)):
					if np.isnan(Z[i, j]):
						label = '--'
						color = 'black'
					else:
						label = f"{Z[i, j]:.2f}"
						color = 'black' if Z[i, j] > 0.6 else 'white'
					ax.text(j, i, label, ha='center', va='center', color=color, fontsize=FONTS_SIZE['cell'])

	# Barra de cores compartilhada
	cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
	cbar = fig.colorbar(im, cax=cbar_ax)
	# 
	cbar.set_label(r"$\Phi$", fontsize=FONTS_SIZE['cbar_label']) # label da barra de cores
	cbar.ax.tick_params(labelsize=FONTS_SIZE['cbar_ticks']) # ticks da barra de cores

	fig.subplots_adjust(
		left=0.06,
		right=0.88,
		top=0.92,
		bottom=0.07,
		wspace=0.12,
		hspace=0.35,
	)

	output_path = os.path.join(output_dir, f"heatmap_phi_{detector_name}_4x3.png")
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close(fig)
	return output_path

SEP_MAIN = "=" * 90
SEP_DETECTOR = "-" * 90
SEP_BLOCK = "." * 90

print(SEP_MAIN)
print(f"RELATORIO DE OSCILACAO | tau={TAU}")
print(SEP_MAIN)

os.makedirs(OUTPUT_DIR, exist_ok=True)
model_reports = {
	backbone: [
		SEP_MAIN,
		f"METRICAS DO MODELO: {backbone}",
		f"TAU={TAU}",
		SEP_MAIN,
	]
	for backbone in backbones
}

for detector in detectors:
	print()
	print(SEP_DETECTOR)
	print(f"DETECTOR: {detector}")
	print(SEP_DETECTOR)
	for backbone in backbones:
		model_reports[backbone].append("")
		model_reports[backbone].append(SEP_DETECTOR)
		model_reports[backbone].append(f"DETECTOR: {detector}")
		model_reports[backbone].append(SEP_DETECTOR)

	df_subset = df_all[(df_all['detector'] == detector)] # Pegar os modelos, i.e., yolov8s, yolo11s, yolo26s
	detector_pair_metrics = {
		backbone: {'phi': [], 'sigma': []}
		for backbone in backbones
	}

	for distance in distances:
		for fluid_height in fluid_heights:
			print(SEP_BLOCK)
			print(f"DISTANCIA: {distance} cm | ALTURA REAL: {fluid_height} mm")
			print(SEP_BLOCK)
			df_subset_pair = df_subset[
				(df_subset['distance'] == distance) &
				(df_subset['true_height'] == fluid_height)
			]
			values = {i: {} for i in backbones}

			for backbone in backbones:
				# Pega a coluna do backbone e calcula media, Phi e sigma por (distance, fluid_height)
				pred_mean = df_subset_pair[backbone].mean()
				phi_global, sigma, n_frames = compute_oscillation_metrics(
					df_subset_pair,
					pred_column=backbone,
					true_column='true_height',
					tau=TAU,
				)
				values[backbone] = {
					'mean': pred_mean,
					'phi': phi_global,
					'sigma': sigma,
					'n': n_frames,
				}

			for backbone in backbones:
				print(
					f"{backbone:<18} | mean={values[backbone]['mean']:.2f} mm | "
					f"Phi={values[backbone]['phi']:.4f} | Sigma={values[backbone]['sigma']:.4f} | "
					f"N={values[backbone]['n']}"
				)
				if not np.isnan(values[backbone]['phi']):
					detector_pair_metrics[backbone]['phi'].append(values[backbone]['phi'])
				if not np.isnan(values[backbone]['sigma']):
					detector_pair_metrics[backbone]['sigma'].append(values[backbone]['sigma'])

				model_reports[backbone].append(
					f"DIST={distance:>2}cm | H={fluid_height:>3}mm | "
					f"mean={values[backbone]['mean']:.2f} | "
					f"Phi={values[backbone]['phi']:.4f} | "
					f"Sigma={values[backbone]['sigma']:.4f} | "
					f"N={values[backbone]['n']}"
				)

	print(SEP_BLOCK)
	print("GLOBAL POR DETECTOR")
	print(SEP_BLOCK)

	for backbone in backbones:
		phi_values = detector_pair_metrics[backbone]['phi']
		sigma_values = detector_pair_metrics[backbone]['sigma']
		phi_global = float(np.mean(phi_values)) if len(phi_values) else np.nan
		sigma = float(np.mean(sigma_values)) if len(sigma_values) else np.nan
		n_pairs = len(phi_values)
		print(f"{backbone:<18} | Phi={phi_global:.4f} | Sigma={sigma:.4f} | N={n_pairs}")
		model_reports[backbone].append(
			f"GLOBAL DETECTOR {detector} | Phi={phi_global:.4f} | Sigma={sigma:.4f} | N={n_pairs}"
		)

	print(SEP_DETECTOR)

print()
print(SEP_MAIN)

for backbone in backbones:
	file_name = model_report_filename(backbone)
	file_path = os.path.join(OUTPUT_DIR, file_name)
	with open(file_path, 'w', encoding='utf-8') as f:
		f.write("\n".join(model_reports[backbone]))
	print(f"Arquivo gerado: {file_path}")

for detector in detectors:
	df_detector = df_all[df_all['detector'] == detector]
	heatmap_path = plot_detector_heatmaps(detector, df_detector, OUTPUT_DIR)
	print(f"Heatmap gerado: {heatmap_path}")