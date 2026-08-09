import json
import platform
import sys
import time
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


def read_frame(capture: cv2.VideoCapture, attempts: int = 30) -> tuple[bool, Any]:
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(0.05)
    return False, None


def open_camera(api: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(0, api)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return capture


def test_single(api: int, requested_name: str) -> dict[str, Any]:
    capture = open_camera(api)
    try:
        opened = capture.isOpened()
        ok, frame = read_frame(capture) if opened else (False, None)
        backend = capture.getBackendName() if opened else None
        return {
            "requested_backend": requested_name,
            "actual_backend": backend,
            "opened": opened,
            "frame_read": ok,
            "frame_shape": list(frame.shape) if ok else None,
        }
    finally:
        capture.release()


def test_media_foundation_sharing() -> dict[str, Any]:
    first = open_camera(cv2.CAP_MSMF)
    second = None
    try:
        first_opened = first.isOpened()
        first_read, _ = read_frame(first) if first_opened else (False, None)
        second = open_camera(cv2.CAP_MSMF)
        second_opened = second.isOpened()
        second_read, _ = read_frame(second) if second_opened else (False, None)
        first_read_after_second, _ = read_frame(first) if first_opened else (False, None)
        return {
            "first_opened": first_opened,
            "first_frame_read": first_read,
            "second_opened": second_opened,
            "second_frame_read": second_read,
            "first_still_reads": first_read_after_second,
            "shared_in_test": all(
                (first_opened, first_read, second_opened, second_read, first_read_after_second)
            ),
        }
    finally:
        if second is not None:
            second.release()
        first.release()


def test_release_and_reopen() -> dict[str, Any]:
    first = open_camera(cv2.CAP_MSMF)
    first_opened = first.isOpened()
    first_read, _ = read_frame(first) if first_opened else (False, None)
    first.release()
    time.sleep(0.5)

    reopened = open_camera(cv2.CAP_MSMF)
    try:
        reopened_ok = reopened.isOpened()
        reopened_read, _ = read_frame(reopened) if reopened_ok else (False, None)
        return {
            "initial_opened": first_opened,
            "initial_frame_read": first_read,
            "reopened": reopened_ok,
            "frame_read_after_reopen": reopened_read,
        }
    finally:
        reopened.release()


def main() -> int:
    # Constructing an Image checks the native MediaPipe image bridge without saving a frame.
    sample = np.zeros((16, 16, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=sample)

    report = {
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "mediapipe": mp.__version__,
            "opencv": cv2.__version__,
        },
        "mediapipe": {
            "image_bridge": mp_image.width == 16 and mp_image.height == 16,
            "face_landmarker_api": hasattr(vision, "FaceLandmarker"),
            "pose_landmarker_api": hasattr(vision, "PoseLandmarker"),
        },
        "camera": {
            "media_foundation": test_single(cv2.CAP_MSMF, "MSMF"),
            "directshow": test_single(cv2.CAP_DSHOW, "DSHOW"),
            "media_foundation_sharing": test_media_foundation_sharing(),
            "release_and_reopen": test_release_and_reopen(),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    mediapipe_ok = all(report["mediapipe"].values())
    camera_ok = report["camera"]["media_foundation"]["frame_read"]
    reopen_ok = report["camera"]["release_and_reopen"]["frame_read_after_reopen"]
    return 0 if mediapipe_ok and camera_ok and reopen_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
