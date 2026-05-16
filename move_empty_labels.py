import os
import shutil

IMG_DIR = "train/images"
LBL_DIR = "train/labels"
OUT_IMG = "train/ignored_empty/images"
OUT_LBL = "train/ignored_empty/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

moved = 0

for lbl in os.listdir(LBL_DIR):
    if not lbl.endswith(".txt"):
        continue

    lbl_path = os.path.join(LBL_DIR, lbl)

    # check if label file is empty
    if os.path.getsize(lbl_path) == 0:
        img_name = lbl.replace(".txt", ".jpg")
        img_path = os.path.join(IMG_DIR, img_name)

        if os.path.exists(img_path):
            shutil.move(img_path, os.path.join(OUT_IMG, img_name))
            shutil.move(lbl_path, os.path.join(OUT_LBL, lbl))
            moved += 1

print(f"✅ Moved {moved} empty-label images")
