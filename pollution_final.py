from ultralytics import YOLO
import cv2
import platform
import os

# ---------------- SOUND ALERT ----------------
def alert_sound():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1200, 500)
    else:
        os.system("echo -e '\a'")

# ---------------- LOAD MODELS ----------------
vehicle_model = YOLO("yolov8n.pt")
smoke_model = YOLO("runs/detect/train2/weights/best.pt")

# ---------------- READ IMAGE ----------------
img = cv2.imread("test.jpg")

if img is None:
    print("❌ ERROR: test.jpg not found")
    exit()

output = img.copy()

# ---------------- VEHICLE DETECTION ----------------
vehicle_results = vehicle_model(img)
vehicle_boxes = []

for r in vehicle_results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = vehicle_model.names[cls]

        if label in ["car", "bus", "truck", "motorcycle"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ---------------- SMOKE DETECTION ----------------
smoke_results = smoke_model(img, conf=0.25)
smoke_boxes = []

for r in smoke_results:
    for box in r.boxes:
        cls = int(box.cls[0])
        label = smoke_model.names[cls]

        if label == "smoke":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            smoke_boxes.append((x1, y1, x2, y2, conf))

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(output, f"SMOKE {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# ---------------- POLLUTION PER VEHICLE ----------------
for vx1, vy1, vx2, vy2 in vehicle_boxes:

    max_conf = 0
    matched = False

    for sx1, sy1, sx2, sy2, conf in smoke_boxes:

        # check overlap
        if (vx1 < sx2 and vx2 > sx1 and vy1 < sy2 and vy2 > sy1):
            matched = True
            max_conf = max(max_conf, conf)

    # ---------------- CLASSIFY ----------------
    if matched:

        if max_conf > 0.6:
            text = "HIGH POLLUTION"
            color = (0, 0, 255)
            alert_sound()
            print("🚨 HIGH POLLUTION DETECTED")

        elif max_conf > 0.4:
            text = "MEDIUM POLLUTION"
            color = (0, 165, 255)
            print("⚠ MEDIUM POLLUTION")

        else:
            text = "LOW POLLUTION"
            color = (0, 255, 255)
            print("🟡 LOW POLLUTION")

    else:
        text = "NO EMISSION"
        color = (255, 255, 0)

    # show label on each vehicle
    cv2.putText(output, text,
                (vx1, vy1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2)

# ---------------- OUTPUT ----------------
cv2.imshow("Pollution Detection System", output)
cv2.imwrite("final_output.jpg", output)

print("✅ Output saved as final_output.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()