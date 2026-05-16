from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.predict(
    source="train/images",
    save=True,
    save_txt=True,     # 🔥 THIS IS THE KEY
    conf=0.25,
    project="runs/detect",
    name="predict_labels"
)

print("✅ Prediction completed with labels")
