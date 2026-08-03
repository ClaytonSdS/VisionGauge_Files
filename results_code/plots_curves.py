import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TEST SET
# ====================================================================================================================================
# REVOCAÇÃO X CONFIANÇA

csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxR_curve\yolov8s_BoxR_curve_test.csv",
	   r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxR_curve\yolov11s_BoxR_curve_test.csv",
	   r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxR_curve\yolov26s_BoxR_curve_test.csv"]

output_name = "boxR_curve.png"
output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxR_curve"
output_path = os.path.join(output_dir, output_name)

xlabel = "Confiança"
ylabel = "Revocação"

# # ====================================================================================================================================
# #PRECISÃO X CONFIANÇA
csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxP_curve\yolov8s_BoxP_curve_test.csv",
        r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxP_curve\yolov11s_BoxP_curve_test.csv",
        r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxP_curve\yolov26s_BoxP_curve_test.csv"]
output_name = "boxP_curve.png"
output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxP_curve"
output_path = os.path.join(output_dir, output_name)
xlabel = "Confiança"
ylabel = "Precisão"
# # ====================================================================================================================================
# # PRECISÃO X REVOCAÇÃO
csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxPR_curve\yolov8s_BoxPR_curve_test.csv",
        r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxPR_curve\yolov11s_BoxPR_curve_test.csv",
        r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxPR_curve\yolov26s_BoxPR_curve_test.csv"]
output_name = "boxPR_curve.png"
output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\boxPR_curve"
output_path = os.path.join(output_dir, output_name)
xlabel = "Precisão"
ylabel = "Revocação"
# # ====================================================================================================================================
# #F1-MEDIDA X CONFIANÇA
csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\f1_curve\yolov8s_BoxF1_curve_test.csv",
        r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\f1_curve\yolov11s_BoxF1_curve_test.csv",
        r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\f1_curve\yolov26s_BoxF1_curve_test.csv"]
output_name = "f1_curve.png"
output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\test\f1_curve"
output_path = os.path.join(output_dir, output_name)
xlabel = "Confiança"
ylabel = "F1-Medida"







# ====================================================================================================================================
# REVOCAÇÃO X CONFIANÇA

# csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxR_curve\yolov8s_BoxR_curve.csv",
# 	   r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxR_curve\yolov11s_BoxR_curve.csv",
# 	   r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxR_curve\yolov26s_BoxR_curve.csv"]

# output_name = "boxR_curve.png"
# output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxR_curve"
# output_path = os.path.join(output_dir, output_name)

# xlabel = "Confiança"
# ylabel = "Revocação"
# # ====================================================================================================================================

# # ====================================================================================================================================
# #PRECISÃO X CONFIANÇA
# csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxP_curve\yolov8s_BoxP_curve.csv",
#         r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxP_curve\yolov11s_BoxP_curve.csv",
#         r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxP_curve\yolov26s_BoxP_curve.csv"]
# output_name = "boxP_curve.png"
# output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxP_curve"
# output_path = os.path.join(output_dir, output_name)
# xlabel = "Confiança"
# ylabel = "Precisão"
# # ====================================================================================================================================
# # PRECISÃO X REVOCAÇÃO
# csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxPR_curve\yolov8s_BoxPR_curve.csv",
#         r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxPR_curve\yolov11s_BoxPR_curve.csv",
#         r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxPR_curve\yolov26s_BoxPR_curve.csv"]
# output_name = "boxPR_curve.png"
# output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\boxPR_curve"
# output_path = os.path.join(output_dir, output_name)
# xlabel = "Precisão"
# ylabel = "Revocação"
# # ====================================================================================================================================
# #F1-MEDIDA X CONFIANÇA
# csvs = [r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\f1_curve\yolov8s_BoxF1_curve.csv",
#         r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\f1_curve\yolov11s_BoxF1_curve.csv",
#         r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\f1_curve\yolov26s_BoxF1_curve.csv"]
# output_name = "f1_curve.png"
# output_dir = r"C:\Users\Clayton\Desktop\vg_rev\VisionGauge_rev\figures\validation\f1_curve"
# output_path = os.path.join(output_dir, output_name)
# xlabel = "Confiança"
# ylabel = "F1-Medida"
# ====================================================================================================================================


# Paleta tableau
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # red: 

legends = ['YOLOv8s', 'YOLO11s', 'YOLO26s']




def _normalize_and_overwrite(path):
	# normaliza CSV: converte colunas numéricas que usam vírgula decimal para float (ponto)
	# e sobrescreve o arquivo com notação de ponto decimal
	df_raw = pd.read_csv(path, dtype=str, header=0)

	# Detecta se o arquivo não tem cabeçalho (pandas usou a primeira linha como header)
	header_is_data = True
	for col in df_raw.columns:
		col_s = str(col).strip()
		# tenta converter nome da coluna para número (substituindo vírgula)
		try:
			float(col_s.replace(',', '.'))
		except Exception:
			header_is_data = False
			break

	if header_is_data:
		# re-leitura sem cabeçalho
		df_raw = pd.read_csv(path, dtype=str, header=None)
		# nomeia as primeiras duas colunas como x,y se houver pelo menos duas
		ncols = df_raw.shape[1]
		if ncols >= 2:
			names = ['x', 'y'] + [f'col{i}' for i in range(2, ncols)]
			df_raw.columns = names
		else:
			df_raw.columns = ['x']

	df_conv = df_raw.copy()
	for col in df_raw.columns:
		s = df_raw[col].astype(str).str.strip()
		# substitui vírgula por ponto para tentar conversão
		conv = pd.to_numeric(s.str.replace(',', '.', regex=False), errors='coerce')
		# se alguma conversão numérica for válida, assume coluna numérica
		if conv.notna().any():
			df_conv[col] = conv
	# salva sobrescrevendo com ponto decimal
	try:
		df_conv.to_csv(path, index=False, float_format='%.6f')
	except Exception:
		# se não puder sobrescrever (permissões), ignore e retorne df_conv
		pass
	return df_conv


def plot_multiple(csv_paths, colors, xlabel=None, ylabel=None, output_path=None, alpha=0.9):
	fig, ax = plt.subplots(figsize=(6, 6))
	plotted = 0

	for i, path in enumerate(csv_paths):
		try:
			df = _normalize_and_overwrite(path)
		except FileNotFoundError:
			print(f'Aviso: arquivo não encontrado: {path}')
			continue
		except pd.errors.EmptyDataError:
			print(f'Aviso: arquivo vazio: {path}')
			continue

		num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
		if len(num_cols) >= 2:
			x = df[num_cols[0]].values
			y = df[num_cols[1]].values
			xlab = num_cols[0]
			ylab = num_cols[1]
		elif len(num_cols) == 1:
			x = np.arange(len(df))
			y = df[num_cols[0]].values
			xlab = 'index'
			ylab = num_cols[0]
		else:
			print(f'Aviso: nenhuma coluna numérica em {path}')
			continue

		color = colors[i % len(colors)] if colors else None
		label = legends[i % len(legends)] if legends else os.path.basename(path).rsplit('.', 1)[0]
		ax.plot(x, y, color=color, linewidth=2, label=label, alpha=alpha)
		plotted += 1

	if plotted == 0:
		raise RuntimeError('Nenhuma curva foi plotada. Verifique os caminhos dos CSVs e o conteúdo.')

	# colocar xticks com fontsize = 10
	ax.tick_params(axis='x', labelsize=12)
	ax.tick_params(axis='y', labelsize=12)
	ax.set_xlabel(xlabel or xlab, fontsize=14)
	ax.set_ylabel(ylabel or ylab, fontsize=14)
	ax.grid(alpha=0.4, linestyle=':')
	ax.legend()

	# limitar eixos X e Y para [0.0, 1.0]
	ax.set_xlim(0.0, 1.0)
	#ax.set_ylim(0.75, 1.0)
	ax.set_ylim(0.0, 1.0)

	# garante que output_path exista; se None, usa arquivo padrão
	if output_path is None:
		output_path = os.path.join(os.getcwd(), 'combined_plot.png')

	out_dir = os.path.dirname(output_path)
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)

	try:
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
	except Exception as e:
		print(f'Erro ao salvar figura em {output_path}: {e}')
		raise
	print(f'Figura salva em: {output_path}')
	plt.show()


def main():
	# garante que o diretório de saída exista e passa output_path
	try:
		os.makedirs(output_dir, exist_ok=True)
	except Exception:
		pass
	plot_multiple(csvs, colors, xlabel=xlabel, ylabel=ylabel, output_path=output_path)


if __name__ == '__main__':
	main()

