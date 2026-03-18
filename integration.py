import time
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
from line_follower import LineFollower

def main():
    # --- 1. Initialization ---
    # Setup YOLO model
    model = YOLO("best.pt")

    # Setup Camera
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    )
    picam2.start()

    # Setup Robot Control (LineFollower)
    lf = LineFollower(
        serial_port="/dev/serial0",
        baud=115200,
        width=640,
        height=480,
        black_thresh=75,    # tune: 60~120
        dead_band=30,       # tune: 30~60
        min_area=900,       # tune: 500~2000
        search_spin=False, # We will handle our own searching logic
        show_debug=True,
    )

    # print("System Ready. Starting detection...")

    try:
        while True:
            # --- 2. Vision Processing ---
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # HSV Masking (as per your code2.py)
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            
            # example: orange ping-pong ball mask
            lower = np.array([5, 120, 120])
            upper = np.array([25, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            
            # optional: apply mask to focus YOLO
            filtered = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

            # --- YOLO on filtered OR original ---
            results = model(filtered, imgsz=416, conf=0.5, verbose=False)
            
            # --- 3. Decision Logic (Find area with most balls) ---
            left_count = 0
            center_count = 0
            right_count = 0

            # Screen split points for 640px width
            # Left: 0-213 | Center: 214-426 | Right: 427-640
            left_boundary = 640 // 3
            right_boundary = (640 // 3) * 2

            for result in results[0].boxes:
                # Get the center x-coordinate of the detection
                box = result.xyxy[0].cpu().numpy()
                center_x = (box[0] + box[2]) / 2

                if center_x < left_boundary:
                    left_count += 1
                elif center_x < right_boundary:
                    center_count += 1
                else:
                    right_count += 1

            # --- 4. Movement Execution ---
            # Determine which zone has the maximum detections
            counts = {"forward": center_count, "left": left_count, "right": right_count}
            target_zone = max(counts, key=counts.get)
            max_balls = counts[target_zone]

            if max_balls > 0:
                print(f"Targeting {target_zone} ({max_balls} balls detected)")
                if target_zone == "forward":
                    lf.send_bool_cmd(1, 0, 0, force=True) # Move Forward
                elif target_zone == "left":
                    lf.send_bool_cmd(0, 1, 0, force=True) # Turn Left
                elif target_zone == "right":
                    lf.send_bool_cmd(0, 0, 1, force=True) # Turn Right
            else:
                # No balls detected: Stop or Spin to search
                print("No targets found. Scanning...")
                lf.send_bool_cmd(0, 0, 0, force=True) # Stop

            # --- 5. Debugging & Display ---
            annotated = results[0].plot()
            # Draw zone boundaries on debug screen
            cv2.line(annotated, (left_boundary, 0), (left_boundary, 480), (255, 0, 0), 2)
            cv2.line(annotated, (right_boundary, 0), (right_boundary, 480), (255, 0, 0), 2)
            
            cv2.imshow("Robot Vision", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        lf.send_bool_cmd(0, 0, 0, force=True)
        lf.close()
        cv2.destroyAllWindows()
        print("Robot Shutdown.")

if __name__ == "__main__":
    main()