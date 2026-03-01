from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np

model = YOLO("best.pt")

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
)
picam2.start()

while True:
    frame = picam2.capture_array()   # RGB888 from camera
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # --- HSV processing ---
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # example: orange ping-pong ball mask
    lower = np.array([5, 120, 120])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # optional: apply mask to focus YOLO
    filtered = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

    # --- YOLO on filtered OR original ---
    results = model(filtered, imgsz=416, conf=0.5, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("YOLO Pi Camera", annotated)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
