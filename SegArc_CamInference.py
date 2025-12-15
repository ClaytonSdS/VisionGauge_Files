import cv2
from ultralytics import YOLO

# Caminhos dos modelos
model_path_1 = "models/SegARC_v00/weights/best.pt"


model_path_1 = "models/SegARC_v02/weights/best.pt"
model_path_2 = "models/SegARC_v04_lr0.0001_5k/weights/best.pt"

# Carrega os modelos
model_1 = YOLO(model_path_1)
model_2 = YOLO(model_path_2)

# Inicializa webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: não foi possível acessar a webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    # Predições
    results_arch1 = model_1.predict(source=frame, conf=0.3, verbose=False)
    results_arch2 = model_2.predict(source=frame, conf=0.3, verbose=False)

    # Frames anotados
    annotated_frame_1 = results_arch1[0].plot()
    annotated_frame_2 = results_arch2[0].plot()

    # Adiciona títulos
    cv2.putText(annotated_frame_1, "MODELO 1 - 1k_augmented", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(annotated_frame_2, "MODELO 2 - 2k_augmented", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Ajusta caso tenham alturas diferentes
    h1, w1, _ = annotated_frame_1.shape
    h2, w2, _ = annotated_frame_2.shape

    if h1 != h2:
        annotated_frame_2 = cv2.resize(annotated_frame_2, (w1, h1))

    # Junta lado a lado
    side_by_side = cv2.hconcat([annotated_frame_1, annotated_frame_2])

    # Limita largura máxima
    max_width = 1280
    h, w = side_by_side.shape[:2]

    if w > max_width:
        scale = max_width / w
        side_by_side = cv2.resize(side_by_side, None, fx=scale, fy=scale)

    # Exibe o resultado lado a lado
    cv2.imshow("Comparação Modelos", side_by_side)

    # Sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finaliza
cap.release()
cv2.destroyAllWindows()
