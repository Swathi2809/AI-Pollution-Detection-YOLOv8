import cv2
import numpy as np

# read image
img = cv2.imread("test.jpg")
output = img.copy()

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define gray/white range (approx smoke)
lower = np.array([0, 0, 120])
upper = np.array([180, 50, 255])

# mask for gray/white
mask = cv2.inRange(hsv, lower, upper)

# blur to merge regions (smoke is diffuse)
mask = cv2.GaussianBlur(mask, (15, 15), 0)

# find contours (possible smoke regions)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 2000:  # filter noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(output, "SMOKE (approx)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# show result
cv2.imshow("Smoke Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save
cv2.imwrite("smoke_output.jpg", output)