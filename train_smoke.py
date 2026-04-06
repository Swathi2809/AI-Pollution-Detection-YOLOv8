from ultralytics import YOLO

# load pretrained model
model = YOLO("yolov8n.pt")

# train on your dataset (FAST SETTINGS)
model.train(
    data="smoke_dataset/data.yaml",
    epochs=3,
    imgsz=320,
    batch=8
)