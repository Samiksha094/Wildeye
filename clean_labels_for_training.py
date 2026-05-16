import os

SRC = "runs/detect/predict/labels"   # change if needed
DST = "train/labels"

os.makedirs(DST, exist_ok=True)

# COCO → YOUR classes
HUMAN = {0}  # person
ANIMALS = {15,16,17,18,19,20,21,22,23,24}  # cat..giraffe
WEAPONS = {43}  # knife (extend later if needed)

kept_files = 0
empty_files = 0
skipped_boxes = 0

for file in os.listdir(SRC):
    if not file.endswith(".txt"):
        continue

    out_lines = []

    with open(os.path.join(SRC, file), "r") as f:
        lines = f.readlines()

    if not lines:
        # empty label file → keep it empty
        open(os.path.join(DST, file), "w").close()
        empty_files += 1
        continue

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
        else:
            skipped_boxes += 1

    if out_lines:
        with open(os.path.join(DST, file), "w") as f:
            f.write("\n".join(out_lines))
        kept_files += 1
    else:
        open(os.path.join(DST, file), "w").close()
        empty_files += 1

print("✅ Files with valid labels:", kept_files)
print("⚠️ Empty label files:", empty_files)
print("❌ Skipped invalid boxes:", skipped_boxes)
