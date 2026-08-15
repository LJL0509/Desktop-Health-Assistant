import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from app_paths import MODEL_ROOT, data_path
from landmark_preview import draw_face, next_video_timestamp_ms, open_camera


FACE_MODEL = MODEL_ROOT / "face_landmarker.task"
SEGMENTER_MODEL = MODEL_ROOT / "selfie_multiclass_256x256.tflite"
DATA_DIR = data_path("upper-body-experiments")
WINDOW_NAME = "Desktop Health Assistant - Upper Body Contour Experiment"
PERSON_CATEGORIES = (2, 4)
HISTORY_SIZE = 90
QUALITY_WINDOW_SIZE = 30
PREPARE_SECONDS = 2.0
RECORD_SECONDS = 6.0
LABEL_KEYS = {
    ord("1"): "normal",
    ord("2"): "whole_body_forward",
    ord("3"): "head_forward",
}


def point(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def centered_segment(xs: np.ndarray, center_x: int) -> tuple[int, int] | None:
    if len(xs) == 0:
        return None
    groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    containing = [group for group in groups if group[0] <= center_x <= group[-1]]
    candidates = containing or groups
    group = min(
        candidates,
        key=lambda item: abs((float(item[0]) + float(item[-1])) / 2.0 - center_x),
    )
    return int(group[0]), int(group[-1])


def visibility_mode(clearance_ratio: float, has_contour: bool) -> str:
    if not has_contour:
        return "CONTOUR UNSTABLE"
    if clearance_ratio < 0.15:
        return "TOO LOW"
    if clearance_ratio < 0.50:
        return "PARTIAL"
    return "FULL"


def stabilized_visibility_mode(
    current: str,
    clearance_ratio: float,
    contour_fraction: float,
) -> str:
    has_contour = contour_fraction >= 0.70
    if current == "TOO LOW" and has_contour and clearance_ratio < 0.18:
        return "TOO LOW"
    if current == "PARTIAL" and has_contour and 0.18 <= clearance_ratio <= 0.55:
        return "PARTIAL"
    if current == "FULL" and has_contour and clearance_ratio >= 0.45:
        return "FULL"
    return visibility_mode(clearance_ratio, has_contour)


def sequence_features(frames: list[dict]) -> dict[str, float] | None:
    baseline = [frame["metrics"] for frame in frames if frame.get("phase") == "baseline" and frame.get("metrics")]
    action = [frame["metrics"] for frame in frames if frame.get("phase") == "action" and frame.get("metrics")]
    if not baseline or not action:
        valid = [frame["metrics"] for frame in frames if frame.get("metrics")]
        if len(valid) < 10:
            return None
        start_count = max(3, len(valid) // 5)
        end_count = max(5, len(valid) * 3 // 10)
        start = valid[:start_count]
        end = valid[-end_count:]
    else:
        if len(baseline) < 3 or len(action) < 5:
            return None
        start = baseline[-min(8, len(baseline)):]
        end = action[-min(8, len(action)):]

    if len(start) < 3 or len(end) < 5:
        return None

    def median(items: list[dict], key: str) -> float:
        return float(np.median([item[key] for item in items]))

    start_face = median(start, "face_width")
    start_torso = median(start, "torso_width")
    return {
        "face_growth": median(end, "face_width") / start_face - 1.0,
        "torso_growth": median(end, "torso_width") / start_torso - 1.0,
        "face_y_change": median(end, "face_center_y") - median(start, "face_center_y"),
        "torso_y_change": median(end, "torso_center_y") - median(start, "torso_center_y"),
        "torso_area_growth": (
            median(end, "torso_area") / median(start, "torso_area") - 1.0
        ),
    }


def classify_motion(features: dict[str, float] | None) -> str:
    if features is None:
        return "INSUFFICIENT"
    face_growth = features["face_growth"]
    torso_growth = features["torso_growth"]
    face_y_change = features["face_y_change"]
    neck_area_loss = features.get("torso_area_growth", 0.0) <= -0.20
    if (
        face_growth >= 0.08
        and torso_growth <= 0.06
        and (face_y_change >= 0.015 or neck_area_loss)
    ):
        return "HEAD FORWARD"
    if face_growth >= 0.12 and torso_growth >= 0.08:
        return "WHOLE BODY FORWARD"
    if abs(face_growth) <= 0.05 and abs(torso_growth) <= 0.07:
        return "STABLE"
    return "UNCERTAIN"


class ExperimentLog:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = DATA_DIR / f"{timestamp}.jsonl"

    def write(self, event: str, **details) -> None:
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            **details,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def upper_body_metrics(
    category_mask: np.ndarray,
    face,
) -> tuple[dict[str, float] | None, np.ndarray]:
    category_mask = np.squeeze(category_mask)
    if category_mask.ndim != 2:
        raise ValueError(f"Expected a 2D category mask, got {category_mask.shape}")
    height, width = category_mask.shape
    left_face = point(face[234], width, height)
    right_face = point(face[454], width, height)
    chin = point(face[152], width, height)
    face_width = abs(right_face[0] - left_face[0])
    if face_width < 20:
        return None, np.zeros_like(category_mask, dtype=np.uint8)

    person = np.where(np.isin(category_mask, PERSON_CATEGORIES), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    person = cv2.morphologyEx(person, cv2.MORPH_CLOSE, kernel)

    x_margin = int(face_width * 2.20)
    x0 = max(0, chin[0] - x_margin)
    x1 = min(width, chin[0] + x_margin)
    y0 = max(0, chin[1] - int(face_width * 0.02))
    y1 = height
    roi = np.zeros_like(person)
    roi[y0:y1, x0:x1] = person[y0:y1, x0:x1]
    roi = largest_component(roi)

    row_data: list[tuple[int, int, int]] = []
    search_start = min(y1, chin[1] + max(2, int(face_width * 0.01)))
    for y in range(search_start, y1 - 3):
        xs = np.flatnonzero(roi[y] > 0)
        segment = centered_segment(xs, chin[0])
        if segment is None:
            continue
        left, right = segment
        span = right - left + 1
        center_x = (left + right) / 2.0
        if not (face_width * 0.40 <= span <= face_width * 3.50):
            continue
        if abs(center_x - chin[0]) > face_width * 0.75:
            continue
        if left <= 2 or right >= width - 3:
            continue
        row_data.append((y, left, right))

    if len(row_data) < max(12, int(face_width * 0.06)):
        return None, roi

    widths = np.array([right - left + 1 for _, left, right in row_data], dtype=float)
    torso_width = float(np.percentile(widths, 75))
    torso_centers = np.array([(left + right) / 2.0 for _, left, right in row_data])
    representative_index = int(np.argmin(np.abs(widths - torso_width)))
    anchor_y, anchor_left, anchor_right = row_data[representative_index]
    sampled_height = row_data[-1][0] - row_data[0][0] + 1
    available_height = max(1, y1 - search_start)
    metrics = {
        "face_width": face_width / width,
        "face_center_x": float(np.mean([item.x for item in face])),
        "face_center_y": float(np.mean([item.y for item in face])),
        "torso_width": torso_width / width,
        "torso_center_x": float(np.median(torso_centers)) / width,
        "torso_center_y": float(np.median([item[0] for item in row_data])) / height,
        "torso_area": float(np.count_nonzero(roi)) / (width * height),
        "face_torso_ratio": face_width / max(torso_width, 1.0),
        "head_clearance_ratio": (height - chin[1]) / face_width,
        "chin_y": chin[1] / height,
        "contour_coverage": sampled_height / available_height,
        "profile_variation": float(np.median(np.abs(widths - np.median(widths)))) / torso_width,
        "contour_top_y": row_data[0][0] / height,
        "contour_bottom_y": row_data[-1][0] / height,
        "anchor_y": anchor_y / height,
        "anchor_left": anchor_left / width,
        "anchor_right": anchor_right / width,
    }
    return metrics, roi


def draw_panel(
    frame: np.ndarray,
    metrics: dict[str, float] | None,
    stability: dict[str, float],
    mode: str,
    clearance_ratio: float,
    message: str,
    recording_label: str | None,
    recording_status: str,
    recording_remaining: float,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (540, 272), (18, 22, 26), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    status = {
        "TOO LOW": "HEAD TOO LOW - RAISE IT SLIGHTLY",
        "CONTOUR UNSTABLE": "UPPER BODY CONTOUR UNSTABLE",
        "PARTIAL": "PARTIAL UPPER BODY DATA",
        "FULL": "FULL UPPER BODY DATA",
    }[mode]
    color = (80, 220, 150) if mode in ("PARTIAL", "FULL") else (70, 110, 245)
    cv2.putText(frame, status, (30, 47), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, color, 2, cv2.LINE_AA)
    rows = (
        ("HEAD CLEARANCE", clearance_ratio),
        ("FACE / TORSO WIDTH", metrics.get("face_torso_ratio", 0.0) if metrics else 0.0),
        ("TORSO WIDTH", metrics.get("torso_width", 0.0) if metrics else 0.0),
        ("CONTOUR COVERAGE", metrics.get("contour_coverage", 0.0) if metrics else 0.0),
        ("RATIO TIME VAR", stability.get("face_torso_ratio", 0.0)),
        ("TORSO TIME VAR", stability.get("torso_width", 0.0)),
    )
    for row, (label, value) in enumerate(rows):
        y = 82 + row * 27
        cv2.putText(frame, label, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (150, 160, 168), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{value:.4f}", (280, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (232, 234, 236), 1, cv2.LINE_AA)
    if recording_label:
        cv2.putText(
            frame,
            f"{recording_status}: {recording_remaining:.1f}s",
            (30, 254),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (40, 220, 245),
            1,
            cv2.LINE_AA,
        )
    elif message:
        cv2.putText(frame, message, (30, 254), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (80, 220, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "1 normal   2 whole body forward   3 head forward   Q quit",
                (18, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 228, 230), 1, cv2.LINE_AA)


def main() -> None:
    missing = [path for path in (FACE_MODEL, SEGMENTER_MODEL) if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing {names}; run scripts/download_models.py")

    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=False,
        output_category_mask=True,
    )

    history: deque[dict[str, float]] = deque(maxlen=HISTORY_SIZE)
    quality_history: deque[tuple[float, bool]] = deque(maxlen=QUALITY_WINDOW_SIZE)
    log = ExperimentLog()
    mode = "CONTOUR UNSTABLE"
    previous_mode = mode
    clearance_ratio = 0.0
    message = ""
    message_until = 0.0
    recording_label: str | None = None
    recording_started = 0.0
    recording_frames: list[dict] = []
    recording_status = ""
    started = time.monotonic()
    video_timestamp_ms = -1
    print("Loading face and segmentation models; the first load can take up to two minutes...", flush=True)
    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.ImageSegmenter.create_from_options(segmenter_options) as segmenter,
    ):
        print("Models loaded. Opening camera...", flush=True)
        camera = open_camera()
        if not camera.isOpened():
            raise RuntimeError("Could not open the camera with Media Foundation.")
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                video_timestamp_ms = next_video_timestamp_ms(
                    time.monotonic() - started,
                    video_timestamp_ms,
                )
                timestamp_ms = video_timestamp_ms
                face_result = face_landmarker.detect_for_video(image, timestamp_ms)
                segment_result = segmenter.segment_for_video(image, timestamp_ms)
                face = face_result.face_landmarks[0] if face_result.face_landmarks else None
                category_mask = segment_result.category_mask.numpy_view()

                metrics = None
                neck_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                if face is not None:
                    height, width = frame.shape[:2]
                    chin = point(face[152], width, height)
                    left_face = point(face[234], width, height)
                    right_face = point(face[454], width, height)
                    face_width = abs(right_face[0] - left_face[0])
                    clearance_ratio = (
                        (height - chin[1]) / face_width
                        if face_width >= 20
                        else 0.0
                    )
                    metrics, neck_mask = upper_body_metrics(category_mask, face)
                    draw_face(frame, face)
                else:
                    clearance_ratio = 0.0

                quality_history.append((clearance_ratio, metrics is not None))
                smooth_clearance = float(np.median([item[0] for item in quality_history]))
                contour_fraction = float(np.mean([item[1] for item in quality_history]))
                mode = stabilized_visibility_mode(
                    previous_mode,
                    smooth_clearance,
                    contour_fraction,
                )
                if mode != previous_mode:
                    history.clear()
                    log.write("visibility_changed", previous=previous_mode, current=mode)
                    previous_mode = mode

                if metrics and mode in ("PARTIAL", "FULL"):
                    history.append(metrics)
                    height, width = frame.shape[:2]
                    left = int(metrics["anchor_left"] * width)
                    right = int(metrics["anchor_right"] * width)
                    y = int(metrics["anchor_y"] * height)
                    cv2.line(frame, (left, y), (right, y), (40, 220, 245), 3, cv2.LINE_AA)
                    cv2.circle(frame, (left, y), 6, (40, 220, 245), -1, cv2.LINE_AA)
                    cv2.circle(frame, (right, y), 6, (40, 220, 245), -1, cv2.LINE_AA)

                tint = np.zeros_like(frame)
                tint[:, :, 1] = neck_mask
                cv2.addWeighted(tint, 0.18, frame, 1.0, 0, frame)
                stability = {}
                if len(history) >= 30:
                    for key in ("face_torso_ratio", "torso_width"):
                        values = np.array([item[key] for item in history], dtype=float)
                        median = float(np.median(values))
                        stability[key] = float(np.median(np.abs(values - median))) / max(abs(median), 1e-6)
                if time.monotonic() >= message_until:
                    message = ""
                now = time.monotonic()
                if recording_label:
                    elapsed = now - recording_started
                    in_prepare = elapsed < PREPARE_SECONDS
                    recording_status = (
                        "HOLD NORMAL"
                        if in_prepare or recording_label == "normal"
                        else f"MOVE NOW: {recording_label.upper()}"
                    )
                    recording_frames.append(
                        {
                            "elapsed_seconds": elapsed,
                            "phase": "baseline" if in_prepare else "action",
                            "mode": mode,
                            "clearance_ratio": clearance_ratio,
                            "metrics": metrics,
                        }
                    )
                    if elapsed >= PREPARE_SECONDS + RECORD_SECONDS:
                        features = sequence_features(recording_frames)
                        prediction = classify_motion(features)
                        log.write(
                            "sequence",
                            label=recording_label,
                            frames=recording_frames,
                            features=features,
                            prediction=prediction,
                        )
                        message = f"PREDICTED {prediction}"
                        message_until = now + 2.0
                        recording_label = None
                        recording_frames = []
                        recording_status = ""
                        history.clear()
                recording_remaining = (
                    max(
                        0.0,
                        (
                            PREPARE_SECONDS - (now - recording_started)
                            if now - recording_started < PREPARE_SECONDS
                            else PREPARE_SECONDS + RECORD_SECONDS - (now - recording_started)
                        ),
                    )
                    if recording_label
                    else 0.0
                )
                draw_panel(
                    frame,
                    metrics,
                    stability,
                    mode,
                    smooth_clearance,
                    message,
                    recording_label,
                    recording_status,
                    recording_remaining,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in LABEL_KEYS:
                    if recording_label:
                        message = "FINISH THE CURRENT RECORDING FIRST"
                    else:
                        recording_label = LABEL_KEYS[key]
                        recording_started = time.monotonic()
                        recording_frames = []
                        recording_status = "HOLD NORMAL"
                        message = ""
                    message_until = time.monotonic() + 2.0
        finally:
            log.write("session_ended")
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
