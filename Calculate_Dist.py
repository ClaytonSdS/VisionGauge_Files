from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


TEST_IMAGES_DIR = Path(r"dataset\UTM_Dataset_Testing.v8-0.9k.folder\images")
VALID_IMAGES_DIR = Path(r"dataset\detector_dataset\valid\images")
OUTPUT_DIR = Path(r"dataset\testing")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root_dir: Path) -> list[Path]:
	return [
		p
		for p in root_dir.rglob("*")
		if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
	]


def mean_histogram_rgb(image_paths: list[Path], bins: int = 256) -> np.ndarray:
	if not image_paths:
		raise ValueError("Nenhuma imagem encontrada para calcular histograma.")

	hist_sum = np.zeros((3, bins), dtype=np.float64)
	valid_count = 0

	for img_path in image_paths:
		img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
		if img is None:
			continue
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		total_pixels = float(img.shape[0] * img.shape[1])
		if total_pixels == 0:
			continue

		for channel_idx in range(3):
			hist = cv2.calcHist([img], [channel_idx], None, [bins], [0, 256]).flatten()
			hist_sum[channel_idx] += hist / total_pixels
		valid_count += 1

	if valid_count == 0:
		raise ValueError("Falha ao ler imagens validas para calcular histograma.")

	return hist_sum / valid_count


def main() -> None:
	project_root = Path(__file__).resolve().parent
	test_dir = (project_root / TEST_IMAGES_DIR).resolve()
	valid_dir = (project_root / VALID_IMAGES_DIR).resolve()
	output_dir = (project_root / OUTPUT_DIR).resolve()
	output_dir.mkdir(parents=True, exist_ok=True)

	if not test_dir.exists():
		raise FileNotFoundError(f"Diretorio de teste nao encontrado: {test_dir}")
	if not valid_dir.exists():
		raise FileNotFoundError(f"Diretorio de validacao nao encontrado: {valid_dir}")

	test_images = list_images(test_dir)
	valid_images = list_images(valid_dir)

	print(f"Imagens de teste encontradas: {len(test_images)}")
	print(f"Imagens de validacao encontradas: {len(valid_images)}")

	hist_test = mean_histogram_rgb(test_images)
	hist_valid = mean_histogram_rgb(valid_images)

	x = np.arange(256)

	fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
	channels = [
		("Canal Vermelho", 0, "#d62728", "#ff9896"),
		("Canal Verde", 1, "#2ca02c", "#98df8a"),
		("Canal Azul", 2, "#1f77b4", "#9ecae1"),
	]

	for ax, (curve_text, idx, color_valid, color_test) in zip(axes, channels):
		ax.plot(
			x,
			hist_valid[idx],
			label="Conjunto de validação",
			linewidth=2.5,
			linestyle="-",
			color=color_valid,
		)
		ax.plot(
			x,
			hist_test[idx],
			label="Conjunto de teste",
			linewidth=1.8,
			linestyle="--",
			marker="o",
			markersize=2.2,
			markevery=16,
			color=color_test,
		)
		ax.set_xlabel("Intensidade do pixel (0-255)")
		ax.grid(linestyle=":", alpha=0.4)
		ax.text(0.03, 0.93, curve_text, transform=ax.transAxes, va="top")
		ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.82))

	axes[0].set_ylabel("Frequência relativa média")
	fig.suptitle(
		"(a) Curvas Do Histograma Médio Por Canal (Conjunto De Teste Vs Conjunto De Validação)",
		x=0.01,
		ha="left",
	)
	plt.tight_layout()

	plot_path = output_dir / "histograma_medio_rgb_test_vs_valid.png"
	plt.savefig(plot_path, dpi=200)
	plt.show()

	csv_path = output_dir / "histograma_medio_rgb_test_vs_valid.csv"
	np.savetxt(
		csv_path,
		np.column_stack(
			(
				x,
				hist_test[0],
				hist_valid[0],
				hist_test[1],
				hist_valid[1],
				hist_test[2],
				hist_valid[2],
			)
		),
		delimiter=",",
		header="bin,r_test,r_valid,g_test,g_valid,b_test,b_valid",
		comments="",
	)

	print(f"Grafico salvo em: {plot_path}")
	print(f"Dados salvos em: {csv_path}")


if __name__ == "__main__":
	main()
