from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image = cv2.imread("test.jpg")

results = model(image)

results[0].show()
results[0].save(filename="output.jpg")