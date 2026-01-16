from ultralytics import YOLO
from PIL import Image
import os
import uuid

# ================= CONFIG =================
MODEL_PATH = r"D:\Trash\test_gan_nhan\refine_last_phase3_10epochs.pt"
INPUT_DIR = r"D:\Trash\test_gan_nhan\unlabeled_images"
OUTPUT_DIR = r"D:\Trash\test_gan_nhan\dataset"
CONF_THRESHOLD = 0.6

# Chỉ auto-label các class bạn muốn
TARGET_CLASSES = {
    "GLASS",
    "PAPER",
    "PLASTIC"
}
# =========================================

model = YOLO(MODEL_PATH)

IMG_OUT = os.path.join(OUTPUT_DIR, "images", "train")
LBL_OUT = os.path.join(OUTPUT_DIR, "labels", "train")
os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

def yolo_format(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h

for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    results = model.predict(
        source=img_path,
        device="cpu",
        conf=CONF_THRESHOLD,
        verbose=False
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print(f"⚠️ Không detect: {file}")
        continue

    new_name = uuid.uuid4().hex[:12]
    img.save(os.path.join(IMG_OUT, f"{new_name}.jpg"))

    label_lines = []

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        class_name = model.names[class_id]

        if class_name not in TARGET_CLASSES:
            continue

        box = boxes.xyxy[i].cpu().numpy()
        x_c, y_c, w, h = yolo_format(box, img_w, img_h)

        label_lines.append(
            f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
        )

    if not label_lines:
        print(f"⚠️ Không class target: {file}")
        continue

    with open(os.path.join(LBL_OUT, f"{new_name}.txt"), "w") as f:
        f.write("\n".join(label_lines))

    os.remove(img_path)
    print(f"✅ {file} → {len(label_lines)} objects")
