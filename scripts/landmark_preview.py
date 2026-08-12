import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
POSE_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
WINDOW_NAME = "Desktop Health Assistant - Landmark Check"

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
FACE_POINTS = (1, 10, 33, 133, 152, 234, 263, 362, 454)


def next_video_timestamp_ms(elapsed_seconds: float, previous_ms: int) -> int:
    measured_ms = int(elapsed_seconds * 1000)
    return max(measured_ms, previous_ms + 1)


def open_camera() -> cv2.VideoCapture:
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def point(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def distance(first, second) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def angle_degrees(first, second) -> float:
    angle = math.degrees(math.atan2(second.y - first.y, second.x - first.x))
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def eye_open_ratio(landmarks, indices: tuple[int, ...]) -> float:
    left, upper_left, upper_right, right, lower_right, lower_left = indices
    horizontal = distance(landmarks[left], landmarks[right])
    vertical = (
        distance(landmarks[upper_left], landmarks[lower_left])
        + distance(landmarks[upper_right], landmarks[lower_right])
    ) / 2.0
    return vertical / horizontal if horizontal > 0 else 0.0


def draw_face(frame: np.ndarray, landmarks) -> dict[str, float]:
    height, width = frame.shape[:2]
    for index in set(LEFT_EYE + RIGHT_EYE + FACE_POINTS):
        cv2.circle(frame, point(landmarks[index], width, height), 2, (80, 220, 150), -1)

    for eye in (LEFT_EYE, RIGHT_EYE):
        eye_points = np.array(
            [point(landmarks[index], width, height) for index in eye], dtype=np.int32
        )
        cv2.polylines(frame, [eye_points], True, (80, 220, 150), 1, cv2.LINE_AA)

    left_ratio = eye_open_ratio(landmarks, LEFT_EYE)
    right_ratio = eye_open_ratio(landmarks, RIGHT_EYE)
    head_tilt = angle_degrees(landmarks[33], landmarks[263])
    face_width = distance(landmarks[234], landmarks[454])
    return {
        "eye_ratio": (left_ratio + right_ratio) / 2.0,
        "head_tilt": head_tilt,
        "face_width": face_width,
    }


def landmark_is_observed(landmark, margin: float = 0.0) -> bool:
    presence = float(getattr(landmark, "presence", 1.0) or 0.0)
    visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
    return (
        presence >= 0.6
        and visibility >= 0.6
        and margin <= landmark.x <= 1.0 - margin
        and margin <= landmark.y <= 1.0 - margin
    )


def shoulders_are_valid(landmarks, face_landmarks) -> bool:
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    if not all(landmark_is_observed(item) for item in (left_shoulder, right_shoulder)):
        return False

    shoulder_span = distance(left_shoulder, right_shoulder)
    if face_landmarks:
        face_width = distance(face_landmarks[234], face_landmarks[454])
        chin_y = face_landmarks[152].y
        return (
            min(left_shoulder.y, right_shoulder.y) > chin_y + 0.03
            and shoulder_span > face_width * 1.05
        )
    return shoulder_span > 0.18


def assess_shoulders(landmarks, face_landmarks) -> tuple[str, str]:
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    shoulders = (left_shoulder, right_shoulder)

    for landmark in shoulders:
        presence = float(getattr(landmark, "presence", 1.0) or 0.0)
        visibility = float(getattr(landmark, "visibility", 1.0) or 0.0)
        if presence < 0.75 or visibility < 0.85:
            return "missing", "low confidence"
        if not (-0.05 <= landmark.x <= 1.05 and 0.0 <= landmark.y <= 1.08):
            return "missing", "outside frame"

    shoulder_span = distance(left_shoulder, right_shoulder)
    if face_landmarks:
        face_width = distance(face_landmarks[234], face_landmarks[454])
        chin_y = face_landmarks[152].y
        if min(left_shoulder.y, right_shoulder.y) <= chin_y + 0.03:
            return "missing", "above chin"
        if shoulder_span <= face_width * 1.05:
            return "missing", "implausible span"
    elif shoulder_span <= 0.18:
        return "missing", "implausible span"

    if all(0.0 <= item.x <= 1.0 and 0.0 <= item.y <= 1.0 for item in shoulders):
        return "found", "fully visible"
    return "partial", "near lower edge"


def draw_pose(frame: np.ndarray, landmarks, face_landmarks) -> dict[str, float | str | None]:
    height, width = frame.shape[:2]
    shoulder_state, shoulder_reason = assess_shoulders(landmarks, face_landmarks)
    if shoulder_state in ("found", "partial"):
        first = landmarks[11]
        second = landmarks[12]
        first_point = point(first, width, height)
        second_point = point(second, width, height)
        first_point = (max(0, min(width - 1, first_point[0])), max(0, min(height - 1, first_point[1])))
        second_point = (max(0, min(width - 1, second_point[0])), max(0, min(height - 1, second_point[1])))
        color = (70, 180, 245) if shoulder_state == "partial" else (80, 220, 150)
        cv2.line(
            frame,
            first_point,
            second_point,
            color,
            2,
            cv2.LINE_AA,
        )
        for item in (first_point, second_point):
            cv2.circle(frame, item, 5, color, -1)

    shoulder_tilt = angle_degrees(landmarks[11], landmarks[12]) if shoulder_state == "found" else None
    return {
        "shoulder_tilt": shoulder_tilt,
        "shoulder_state": shoulder_state,
        "shoulder_reason": shoulder_reason,
        "shoulders_visible": float(shoulder_state == "found"),
        "shoulders_partial": float(shoulder_state == "partial"),
    }


def draw_panel(frame: np.ndarray, metrics: dict[str, float | str | None], fps: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (310, 180), (18, 22, 26), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    face_found = bool(metrics.get("face_found", 0.0))
    shoulder_state = str(metrics.get("shoulder_state", "missing"))
    shoulder_tilt = metrics.get("shoulder_tilt")
    lines = (
        ("FACE", "FOUND" if face_found else "MISSING"),
        ("SHOULDERS", shoulder_state.upper()),
        ("EYE OPEN RATIO", f"{metrics.get('eye_ratio', 0.0):.3f}"),
        ("HEAD TILT", f"{metrics.get('head_tilt', 0.0):+.1f} deg"),
        ("SHOULDER TILT", "N/A" if shoulder_tilt is None else f"{shoulder_tilt:+.1f} deg"),
        ("FACE WIDTH", f"{metrics.get('face_width', 0.0):.3f}"),
    )
    for row, (label, value) in enumerate(lines):
        y = 42 + row * 22
        cv2.putText(frame, label, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                    (155, 164, 172), 1, cv2.LINE_AA)
        color = (80, 220, 150) if (face_found and shoulder_state == "found") else (70, 180, 245)
        cv2.putText(frame, value, (172, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 90, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 228, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, "P pause/release    Q quit", (20, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 232, 234), 1, cv2.LINE_AA)


def paused_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), (22, 26, 30), dtype=np.uint8)
    cv2.putText(frame, "CAMERA RELEASED", (185, 220), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (80, 220, 150), 2, cv2.LINE_AA)
    cv2.putText(frame, "Press P to resume", (222, 258), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (175, 182, 188), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    missing = [str(path) for path in (FACE_MODEL, POSE_MODEL) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Model files are missing. Run: python scripts/download_models.py"
        )

    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = open_camera()
    paused = False
    started = time.monotonic()
    video_timestamp_ms = -1
    previous_frame_time = started
    smoothed_fps = 0.0

    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            while True:
                if paused:
                    frame = paused_frame()
                else:
                    ok, frame = camera.read()
                    if not ok:
                        camera.release()
                        paused = True
                        frame = paused_frame()
                    else:
                        frame = cv2.flip(frame, 1)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb = np.ascontiguousarray(rgb)
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        video_timestamp_ms = next_video_timestamp_ms(
                            time.monotonic() - started,
                            video_timestamp_ms,
                        )
                        timestamp_ms = video_timestamp_ms

                        face_result = face_landmarker.detect_for_video(image, timestamp_ms)
                        pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
                        metrics: dict[str, float] = {
                            "face_found": float(bool(face_result.face_landmarks)),
                            "pose_found": float(bool(pose_result.pose_landmarks)),
                        }
                        face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
                        if face_landmarks:
                            metrics.update(draw_face(frame, face_landmarks))
                        if pose_result.pose_landmarks:
                            metrics.update(draw_pose(frame, pose_result.pose_landmarks[0], face_landmarks))

                        now = time.monotonic()
                        instant_fps = 1.0 / max(now - previous_frame_time, 1e-6)
                        smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
                        previous_frame_time = now
                        draw_panel(frame, metrics, smoothed_fps)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    if paused:
                        camera = open_camera()
                        paused = not camera.isOpened()
                        previous_frame_time = time.monotonic()
                    else:
                        camera.release()
                        paused = True
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
