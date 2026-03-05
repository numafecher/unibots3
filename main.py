import time
import cv2
from line_follower import LineFollower


def main():
    lf = LineFollower(
        serial_port="/dev/serial0",  # if USB instead, it might be /dev/ttyUSB0
        baud=115200,
        width=640,
        height=480,
        black_thresh=75,   # tune: 60~120
        dead_band=30,      # tune: 30~60
        min_area=900,      # tune: 500~2000
        search_spin=True,
        show_debug=True,
    )

    try:
        while True:
            F, L, R = lf.step()

            # Add your extra "main logic" here later
            # Example override:
            # if something_happens:
            #     lf.send_bool_cmd(0,0,0, force=True)

            time.sleep(0.01)

            if lf.show_debug:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        lf.close()


if __name__ == "__main__":
    main()
