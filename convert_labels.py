import os

SRC = "runs/detect/predict_labels/labels"
DST = "train/labels"

os.makedirs(DST, exist_ok=True)

# COCO → YOUR CLASSES
HUMAN = {0}   # person
ANIMALS = {15,16,17,18,19,20,21,22,23,24,28}  # animals
WEAPONS = {43,67}  # knife / weapon-like

converted = 0
empty = 0

for file in os.listdir(SRC):
    if not file.endswith(".txt"):
        continue

    out_lines = []

    with open(os.path.join(SRC, file)) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, x, y, w, h = parts
        cls = int(cls)

        if cls in HUMAN:
            out_lines.append(f"0 {x} {y} {w} {h}")
        elif cls in WEAPONS:
            out_lines.append(f"1 {x} {y} {w} {h}")
        elif cls in ANIMALS:
            out_lines.append(f"2 {x} {y} {w} {h}")

    with open(os.path.join(DST, file), "w") as f:
        f.write("\n".join(out_lines))

    if out_lines:
        converted += 1
    else:
        empty += 1

print("✅ Converted files:", converted)
print("⚠️ Empty label files:", empty)
