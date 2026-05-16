from ultralytics import YOLO

# Load best model from previous training
model = YOLO("runs/detect/train3/weights/best.pt")

model.train(
    data="data.yaml",
    epochs=2,          # fine-tuning only
    imgsz=320,          # helps weapon detection
    batch=8,
    workers=2
)
