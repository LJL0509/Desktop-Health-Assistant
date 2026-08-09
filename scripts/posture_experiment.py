import json
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from landmark_preview import (
    distance,
    draw_face,
    open_camera,
    point,
)


ROOT = Path(__file__).resolve().parents[1]
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
POSE_MODEL = ROOT / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR = ROOT / "data" / "posture-experiments"
WINDOW_NAME = "Desktop Health Assistant - Posture Experiment"
RECORD_SECONDS = 10.0
PREPARE_SECONDS = 3.0
LABELS = {
    ord("1"): "normal",
    ord("2"): "head_forward",
    ord("3"): "whole_body_forward",
}


def midpoint(first, second, axis: str) -> float:
    return (float(getattr(first, axis)) + float(getattr(second, axis))) / 2.0


def neck_anchor_mode(face, pose_image, pose_world) -> tuple[str | None, str]:
    if not face:
        return None, "face_missing"
    if not pose_image:
        return None, "pose_missing"
    if not pose_world:
        return None, "pose_world_missing"

    left_shoulder, right_shoulder = pose_image[11], pose_image[12]
    for name, landmark in (("left", left_shoulder), ("right", right_shoulder)):
        values = (landmark.x, landmark.y, pose_world[11 if name == "left" else 12].z)
        if not all(np.isfinite(value) for value in values):
            return None, f"{name}_anchor_not_finite"
        if not (-0.20 <= landmark.x <= 1.20 and 0.0 <= landmark.y <= 1.50):
            return None, f"{name}_anchor_too_far"

    shoulder_width = distance(left_shoulder, right_shoulder)
    face_width = distance(face[234], face[454])
    chin_y = face[152].y
    if min(left_shoulder.y, right_shoulder.y) <= chin_y + 0.03:
        return None, "anchor_not_below_chin"
    if not (face_width * 1.05 < shoulder_width < face_width * 4.0):
        return None, "anchor_span_implausible"

    depth_asymmetry = abs(pose_world[11].z - pose_world[12].z)
    if depth_asymmetry > 0.20:
        return None, "body_turned_sideways"

    fully_visible = all(
        0.0 <= item.x <= 1.0 and 0.0 <= item.y <= 1.0
        for item in (left_shoulder, right_shoulder)
    )
    min_presence = min(float(item.presence or 0.0) for item in (left_shoulder, right_shoulder))
    min_visibility = min(float(item.visibility or 0.0) for item in (left_shoulder, right_shoulder))
    direct = fully_visible and min_presence >= 0.60 and min_visibility >= 0.60
    return ("direct" if direct else "estimated"), "valid"


def posture_metrics(face, pose_image, pose_world) -> tuple[dict[str, float] | None, str]:
    anchor_mode, reason = neck_anchor_mode(face, pose_image, pose_world)
    if anchor_mode is None:
        return None, reason

    left_shoulder_2d, right_shoulder_2d = pose_image[11], pose_image[12]
    shoulder_width_2d = distance(left_shoulder_2d, right_shoulder_2d)
    face_width_2d = distance(face[234], face[454])

    left_ear_2d, right_ear_2d = pose_image[7], pose_image[8]
    left_ear_3d, right_ear_3d = pose_world[7], pose_world[8]
    left_shoulder_3d, right_shoulder_3d = pose_world[11], pose_world[12]

    ear_y = midpoint(left_ear_2d, right_ear_2d, "y")
    shoulder_y = midpoint(left_shoulder_2d, right_shoulder_2d, "y")
    ear_z = midpoint(left_ear_3d, right_ear_3d, "z")
    shoulder_z = midpoint(left_shoulder_3d, right_shoulder_3d, "z")

    if shoulder_width_2d <= 1e-6:
        return None, "shoulder_width_zero"

    return {
        "head_shoulder_depth": ear_z - shoulder_z,
        "face_shoulder_ratio": face_width_2d / shoulder_width_2d,
        "neck_vertical_ratio": (shoulder_y - ear_y) / shoulder_width_2d,
        "face_width": face_width_2d,
        "shoulder_width": shoulder_width_2d,
        "shoulder_mid_y": shoulder_y,
        "shoulder_depth_asymmetry": abs(left_shoulder_3d.z - right_shoulder_3d.z),
        "head_lateral_offset": (
            midpoint(left_ear_2d, right_ear_2d, "x")
            - midpoint(left_shoulder_2d, right_shoulder_2d, "x")
        ) / shoulder_width_2d,
        "anchor_estimated": float(anchor_mode == "estimated"),
        "anchor_min_presence": min(
            float(left_shoulder_2d.presence or 0.0),
            float(right_shoulder_2d.presence or 0.0),
        ),
        "anchor_min_visibility": min(
            float(left_shoulder_2d.visibility or 0.0),
            float(right_shoulder_2d.visibility or 0.0),
        ),
    }, "valid"


def draw_neck_anchor(frame: np.ndarray, pose_image, estimated: bool) -> None:
    height, width = frame.shape[:2]
    left = point(pose_image[11], width, height)
    right = point(pose_image[12], width, height)
    left = (max(0, min(width - 1, left[0])), max(0, min(height - 1, left[1])))
    right = (max(0, min(width - 1, right[0])), max(0, min(height - 1, right[1])))
    color = (70, 180, 245) if estimated else (80, 220, 150)
    cv2.line(frame, left, right, color, 2, cv2.LINE_AA)
    for item in (left, right):
        cv2.circle(frame, item, 5, color, -1)


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    keys = [key for key in samples[0] if key != "elapsed_seconds"]
    return {
        key: float(np.median([sample[key] for sample in samples]))
        for key in keys
    }


def summarize_mad(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}
    keys = [key for key in samples[0] if key != "elapsed_seconds"]
    result: dict[str, float] = {}
    for key in keys:
        values = np.asarray([sample[key] for sample in samples], dtype=np.float64)
        median = np.median(values)
        result[key] = float(np.median(np.abs(values - median)))
    return result


def save_recording(
    label: str,
    samples: list[dict[str, float]],
    invalid_counts: Counter[str],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = OUTPUT_DIR / f"{timestamp}-{label}.json"
    payload = {
        "label": label,
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "record_seconds": RECORD_SECONDS,
        "valid_sample_count": len(samples),
        "invalid_frame_counts": dict(invalid_counts),
        "frames_saved": False,
        "summary_median": summarize(samples),
        "summary_mad": summarize_mad(samples),
        "samples": samples,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def draw_experiment_panel(
    frame: np.ndarray,
    metrics: dict[str, float] | None,
    pending_label: str | None,
    pending_started: float,
    active_label: str | None,
    recording_started: float,
    last_message: str,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (430, 236), (18, 22, 26), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    cv2.putText(frame, "POSTURE DATA CHECK", (30, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (238, 240, 242), 2, cv2.LINE_AA)
    valid = metrics is not None
    anchor_label = "DIRECT" if metrics and metrics.get("anchor_estimated", 0.0) < 0.5 else "ESTIMATED"
    status = f"VALID NECK ANCHOR: {anchor_label}" if valid else "INVALID NECK ANCHOR"
    cv2.putText(frame, status, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 150) if valid else (70, 180, 245),
                1, cv2.LINE_AA)

    rows = (
        ("HEAD-SHOULDER Z", "head_shoulder_depth"),
        ("FACE / SHOULDER", "face_shoulder_ratio"),
        ("NECK VERTICAL", "neck_vertical_ratio"),
        ("SHOULDER WIDTH", "shoulder_width"),
        ("L/R DEPTH DIFF", "shoulder_depth_asymmetry"),
    )
    for row, (label, key) in enumerate(rows):
        y = 98 + row * 24
        value = metrics.get(key, 0.0) if metrics else 0.0
        cv2.putText(frame, label, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (150, 160, 168), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{value:+.4f}", (250, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (232, 234, 236), 1, cv2.LINE_AA)

    now = time.monotonic()
    if pending_label:
        remaining = max(0.0, PREPARE_SECONDS - (now - pending_started))
        status = f"PREPARE {pending_label}: {remaining:.1f}s"
    elif active_label:
        remaining = max(0.0, RECORD_SECONDS - (now - recording_started))
        status = f"RECORDING {active_label}: {remaining:.1f}s"
    else:
        status = last_message or "1 normal   2 head forward   3 whole body forward"
    cv2.putText(frame, status, (24, frame.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (80, 220, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q quit", (frame.shape[1] - 75, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 224, 226), 1, cv2.LINE_AA)


def main() -> None:
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    camera = open_camera()
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera with Media Foundation.")

    session_started = time.monotonic()
    pending_label: str | None = None
    pending_started = 0.0
    active_label: str | None = None
    recording_started = 0.0
    samples: list[dict[str, float]] = []
    invalid_counts: Counter[str] = Counter()
    last_message = ""

    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame read failed.")
                frame = cv2.flip(frame, 1)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.monotonic() - session_started) * 1000)

                face_result = face_landmarker.detect_for_video(image, timestamp_ms)
                pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
                face = face_result.face_landmarks[0] if face_result.face_landmarks else None
                pose_image = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                pose_world = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None
                metrics, validity_reason = posture_metrics(face, pose_image, pose_world)

                if face:
                    draw_face(frame, face)
                if metrics and pose_image:
                    draw_neck_anchor(frame, pose_image, metrics["anchor_estimated"] >= 0.5)

                now = time.monotonic()
                if pending_label and now - pending_started >= PREPARE_SECONDS:
                    active_label = pending_label
                    pending_label = None
                    recording_started = now
                    samples = []
                    invalid_counts = Counter()

                if active_label:
                    if metrics:
                        samples.append({"elapsed_seconds": now - recording_started, **metrics})
                    else:
                        invalid_counts[validity_reason] += 1
                    if now - recording_started >= RECORD_SECONDS:
                        destination = save_recording(active_label, samples, invalid_counts)
                        last_message = f"SAVED {destination.name} ({len(samples)} valid)"
                        active_label = None
                        samples = []

                draw_experiment_panel(
                    frame,
                    metrics,
                    pending_label,
                    pending_started,
                    active_label,
                    recording_started,
                    last_message,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in LABELS and not pending_label and not active_label:
                    pending_label = LABELS[key]
                    pending_started = time.monotonic()
                    last_message = ""
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
