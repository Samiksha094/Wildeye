import os
import shutil

SRC = "runs/detect/predict/labels"
DST = "train/labels"

os.makedirs(DST, exist_ok=True)

count = 0
for f in os.listdir(SRC):
    if f.endswith(".txt"):
        shutil.copy(
            os.path.join(SRC, f),
            os.path.join(DST, f)
        )
        count += 1

print(f"✅ Copied {count} labels into train/labels")
