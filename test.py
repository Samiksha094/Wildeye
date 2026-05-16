from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

model.predict(
    source="test/images",
    conf=0.45,
    iou=0.5,
    save=True
)
