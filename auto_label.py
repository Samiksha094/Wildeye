from ultralytics import YOLO
import os

# Load pretrained model (file is already in your project)
model = YOLO("yolov8m.pt")

# Paths (CORRECT for your folder structure)
IMG_DIR = "train/images"
LBL_DIR = "train/labels"

os.makedirs(LBL_DIR, exist_ok=True)

# Auto-label
model.predict(
    source=IMG_DIR,
    save=False,
    save_txt=True,
    conf=0.3
)

print("✅ Auto-labeling completed for train/images")
