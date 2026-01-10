from ultralytics import YOLO
from PIL import Image
import os
import shutil
import uuid

# ================= CONFIG =================
MODEL_PATH = r"D:\Trash\test_gan_nhan\refine_last_phase3_10epochs.pt"
INPUT_DIR = r"D:\Trash\test_gan_nhan\unlabeled_images"
OUTPUT_DIR = r"D:\Trash\test_gan_nhan\data"
CONF_THRESHOLD = 0.4

# Map class name → folder
CLASS_FOLDER_MAP = {
    "GLASS": "glass",
    "PAPER": "paper",
    "PLASTIC": "plastic"
}
# =========================================

model = YOLO(MODEL_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def yolo_format(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h


for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
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

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"⚠️ Không detect: {file}")
        continue

    # === Lấy box có confidence cao nhất ===
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax().item()

    box = boxes.xyxy[best_idx].cpu().numpy()
    score = boxes.conf[best_idx].item()
    class_id = int(boxes.cls[best_idx].item())
    class_name = model.names[class_id]

    if class_name not in CLASS_FOLDER_MAP:
        print(f"⚠️ Bỏ qua class: {class_name}")
        continue

    # === Tạo tên mới ===
    new_name = uuid.uuid4().hex[:12]
    new_img_name = f"{new_name}.jpg"
    new_label_name = f"{new_name}.txt"

    # === Tạo thư mục ===
    class_dir = CLASS_FOLDER_MAP[class_name]
    img_out_dir = os.path.join(OUTPUT_DIR, class_dir, "image")
    label_out_dir = os.path.join(OUTPUT_DIR, class_dir, "label")

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    # === Save image ===
    new_img_path = os.path.join(img_out_dir, new_img_name)
    img.save(new_img_path)

    # === Save label ===
    x_c, y_c, w, h = yolo_format(box, img_w, img_h)
    label_path = os.path.join(label_out_dir, new_label_name)

    with open(label_path, "w") as f:
        f.write(f"{x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    # === Remove original image ===
    os.remove(img_path)

    print(f"✅ {file} → {class_name} ({score:.2f})")
