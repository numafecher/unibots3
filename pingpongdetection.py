from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
#import smbus

#bus = smbus.SMBus(1)
#ESP32_ADDR = 0x08

model = YOLO("best.pt")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

while True:
    frame = picam2.capture_array()

    # If frame is BGRA/RGBA (4 channels), convert to BGR (3 channels)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #COLOR_BGRA2BGR

    results = model(frame, imgsz=416, conf=0.5)
    # - I2C
    # Check if any boxes were detected
    #if len(results[0].boxes) > 0:
        # Get the Class ID of the highest-confidence detection
        # Class 0 is usually 'person' in the COCO dataset
        #class_id = int(results[0].boxes.cls[0].item())
        
        #try:
            # Send the numeric class ID to the ESP32
            #bus.write_byte(ESP32_ADDR, class_id)
        #except Exception as e:
            # This prevents the script from crashing if a wire jiggles loose
            #print(f"I2C Communication Error: {e}")
    # - End 
    
    annotated = results[0].plot()

    cv2.imshow("YOLO Pi Camera", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
