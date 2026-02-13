import cv2
from VisionGauge import VisionGauge

model  = VisionGauge()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#cap = cv2.VideoCapture("http://192.168.2.117:8080/video")

model.predict_streaming(
    camera,
    frame_height=1280,
    frame_width=720,
    frame_thickness=4,
    frame_color="#551bb3",
    font_color="#ffffff",
    fontsize=10
)
