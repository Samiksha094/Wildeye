from ultralytics import YOLO

# Load model
model = YOLO("yolov8m.pt")

# Run prediction
model.predict(
    source="train/images",
    save=True,
    conf=0.3
)

print("✅ Prediction completed")
