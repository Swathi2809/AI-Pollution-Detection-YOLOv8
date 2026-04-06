
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

results = model.val(data="smoke_dataset/data.yaml")

print(results)