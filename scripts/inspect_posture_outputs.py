from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("stage=camera_open", flush=True)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        ok, frame = False, None
        for _ in range(30):
            ok, frame = camera.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        if not ok:
            print("camera_frame=false")
            return 1
    finally:
        camera.release()

    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    print("stage=model_initialize", flush=True)

    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "face_landmarker.task")),
        output_facial_transformation_matrixes=True,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "pose_landmarker_lite.task")),
    )

    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        print("stage=inference", flush=True)
        face = face_landmarker.detect(image)
        pose = pose_landmarker.detect(image)

    print(f"face_detected={bool(face.face_landmarks)}")
    print(f"face_transform={bool(face.facial_transformation_matrixes)}")
    print(f"pose_detected={bool(pose.pose_landmarks)}")
    print(f"pose_world_landmarks={bool(pose.pose_world_landmarks)}")

    if pose.pose_world_landmarks:
        points = pose.pose_world_landmarks[0]
        for name, index in (("left_ear", 7), ("right_ear", 8), ("left_shoulder", 11), ("right_shoulder", 12)):
            item = points[index]
            print(f"{name}=x:{item.x:+.4f},y:{item.y:+.4f},z:{item.z:+.4f}")

    print("frames_saved=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
