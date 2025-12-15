import cv2
import os
import tqdm

video_path = r"folder_test\video.mp4"
output_dir = r"folder_test\imgs"
os.makedirs(output_dir, exist_ok=True)

def crop_bottom_center(
    img,
    crop_h=1500,
    crop_w=1500,
    horizontal_shift=0,
    start_row=None
):
    height, width, _ = img.shape
    cx = width // 2 + horizontal_shift
    cx = max(crop_w // 2, min(width - crop_w // 2, cx))
    if start_row is None:
        start_row = height - crop_h
    start_row = max(0, min(height - crop_h, start_row))
    end_row = start_row + crop_h
    start_col = cx - crop_w // 2
    end_col = cx + crop_w // 2
    return img[start_row:end_row, start_col:end_col]

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

skip = 9
frame_index = 0
saved_index = 4723

for _ in tqdm.tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    # pula frames
    if frame_index % skip != 0:
        frame_index += 1
        continue

    # processa somente os frames escolhidos
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    crop = crop_bottom_center(
        frame_rgb,
        crop_h=1000,
        crop_w=1000,
        start_row=0,
        horizontal_shift=0
    )

    save_path = os.path.join(output_dir, f"image_{saved_index}.png")
    cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    saved_index += 1
    frame_index += 1

cap.release()
print("Concluído!")
