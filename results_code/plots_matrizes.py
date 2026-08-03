import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================================================================================================================
# MATRIZES CONFUSAO CONJUNTO DE VALIDAÇÃO (VAL)
matrix_yolov8s = np.array([[1761, 9],
						   [2, None]])    # fluid-interface x background

matrix_yolov11s = np.array([[1759, 6],
							[4, None]])    # fluid-interface x background

matrix_yolov26s = np.array([[1759, 17],
							[4, None]])    # fluid-interface x background

output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\matrizes"


# ====================================================================================================================================
# MATRIZES CONFUSAO CONJUNTO DE TESTE (TEST) 
output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\matrizes"

matrix_yolov8s = np.array([[903, None],
						   [1, None]])    # fluid-interface x background

matrix_yolov11s = np.array([[904, 4],
							[None, None]])    # fluid-interface x background

matrix_yolov26s = np.array([[904, 7],
							[None, None]])    # fluid-interface x background





xlabel = "Classe Verdadeira"
ylabel = "Classe Prevista"
# ====================================================================================================================================


# Paleta tableau
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # red: 
legends = ['YOLOv8s', 'YOLOv11s', 'YOLOv26s']

fontsize = 6


def plot_confusion(matrix, cmap_name, title, output_path, class_names=None):
	# converte elementos None para np.nan e garante float
	mat = np.array([[float(x) if x is not None else np.nan for x in row] for row in matrix], dtype=float)

	if class_names is None:
		class_names = ['Fluid-interface', 'Background']

	vmax = np.nanmax(mat) if not np.isnan(np.nanmax(mat)) else 1.0

	fig, ax = plt.subplots(figsize=(4, 4))
	im = ax.imshow(mat, cmap=cmap_name, vmin=0.0, vmax=vmax)

	# anota os valores dentro das células (fonte menor que 10)
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			val = mat[i, j]
			# valores None ficam vazios (string vazia)
			txt = '' if np.isnan(val) else str(int(val))
			# primeiro elemento [0,0] sempre com texto branco
			if i == 0 and j == 0:
				text_color = 'white'
			else:
				text_color = 'black'
			ax.text(j, i, txt, ha='center', va='center', color=text_color, fontsize=fontsize)

	# ticks com fonte menor que 10
	ax.set_xticks(np.arange(len(class_names)))
	ax.set_yticks(np.arange(len(class_names)))
	ax.set_xticklabels(class_names, fontsize=fontsize)
	ax.set_yticklabels(class_names, fontsize=fontsize)

	# labels maiores (tamanho 10)
	ax.set_xlabel('Classe Verdadeira', fontsize=6)
	ax.set_ylabel('Classe Prevista', fontsize=6)

	# só define título se passado (permite "sem título")
	if title:
		ax.set_title(title)

	# remover a moldura externa ao redor da matriz
	for spine in ax.spines.values():
		spine.set_visible(False)
	cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	# reduzir fonte dos labels do colorbar e remover moldura (outline)
	cbar.ax.tick_params(labelsize=fontsize)
	try:
		cbar.outline.set_visible(False)
	except Exception:
		pass

	out_dir = os.path.dirname(output_path)
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)

	plt.savefig(output_path, dpi=300, bbox_inches='tight')
	plt.close(fig)
	print(f'Salvo: {output_path}')


def main():
	# plota cada matriz com a paleta solicitada
	plot_confusion(matrix_yolov8s, 'Blues', '', os.path.join(output_dir, 'matrix_yolov8s.png'))
	# trocar colormaps: YOLOv11s -> Oranges, YOLOv26s -> Greens
	plot_confusion(matrix_yolov11s, 'Oranges', '', os.path.join(output_dir, 'matrix_yolov11s.png'))
	plot_confusion(matrix_yolov26s, 'Greens', '', os.path.join(output_dir, 'matrix_yolov26s.png'))


if __name__ == '__main__':
	main()

