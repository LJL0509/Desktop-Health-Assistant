import time

import cv2


WINDOW_NAME = "Desktop Health Assistant - Camera Field of View"
MODES = {
    ord("1"): (640, 480),
    ord("2"): (1280, 720),
    ord("3"): (1920, 1080),
}


def open_camera(width: int, height: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def main() -> None:
    requested = MODES[ord("1")]
    camera = open_camera(*requested)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera with Media Foundation.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera frame read failed.")
            frame = cv2.flip(frame, 1)
            actual_height, actual_width = frame.shape[:2]

            overlay = frame.copy()
            panel_width = min(actual_width - 20, 500)
            cv2.rectangle(overlay, (12, 12), (panel_width, 104), (18, 22, 26), -1)
            cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
            cv2.putText(
                frame,
                f"REQUESTED {requested[0]} x {requested[1]}",
                (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (235, 238, 240),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"ACTUAL {actual_width} x {actual_height}",
                (28, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (80, 220, 150),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "1 640x480   2 1280x720   3 1920x1080   Q quit",
                (28, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (170, 178, 184),
                1,
                cv2.LINE_AA,
            )

            center_x, center_y = actual_width // 2, actual_height // 2
            cv2.line(frame, (center_x, 0), (center_x, actual_height), (80, 220, 150), 1)
            cv2.line(frame, (0, center_y), (actual_width, center_y), (80, 220, 150), 1)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in MODES and MODES[key] != requested:
                requested = MODES[key]
                camera.release()
                time.sleep(0.3)
                camera = open_camera(*requested)
                if not camera.isOpened():
                    raise RuntimeError(f"Camera mode {requested} could not be opened.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
