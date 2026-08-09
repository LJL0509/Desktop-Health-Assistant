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

from landmark_preview import draw_face, open_camera
from posture_experiment import FACE_MODEL, POSE_MODEL, posture_metrics


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "monitor-sessions"
WINDOW_NAME = "Desktop Health Assistant - Neck Monitor Prototype"
CALIBRATION_SECONDS = 15.0
WINDOW_SECONDS = 5.0
PREPARE_SECONDS = 3.0
STATE_CONFIRM_SECONDS = 3.0
DATA_ALERT_SECONDS = 10.0


def median_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in samples[0] if key != "elapsed_seconds"]
    return {
        key: float(np.median([sample[key] for sample in samples]))
        for key in keys
    }


def mad_metrics(samples: list[dict[str, float]], medians: dict[str, float]) -> dict[str, float]:
    return {
        key: float(np.median([abs(sample[key] - medians[key]) for sample in samples]))
        for key in medians
    }


def build_baseline(samples: list[dict[str, float]]) -> tuple[dict | None, str]:
    if len(samples) < 100:
        return None, "not enough valid data"
    medians = median_metrics(samples)
    mads = mad_metrics(samples, medians)
    stability = {
        key: mads[key] / max(abs(medians[key]), 1e-6)
        for key in ("face_width", "shoulder_width", "face_shoulder_ratio")
    }
    if stability["face_width"] > 0.06:
        return None, "face scale was unstable"
    if stability["shoulder_width"] > 0.10:
        return None, "virtual anchor was unstable"
    if stability["face_shoulder_ratio"] > 0.10:
        return None, "face/body ratio was unstable"
    return {
        "median": medians,
        "mad": mads,
        "relative_mad": stability,
    }, "ok"


def thresholds_from_baseline(baseline: dict) -> dict[str, float]:
    median = baseline["median"]
    mad = baseline["mad"]
    return {
        "face_growth": max(0.08, 3.0 * mad["face_width"] / median["face_width"]),
        "body_growth": max(0.10, 3.0 * mad["shoulder_width"] / median["shoulder_width"]),
        "ratio_growth": max(
            0.08,
            2.5 * mad["face_shoulder_ratio"] / median["face_shoulder_ratio"],
        ),
    }


def classify_posture(baseline: dict, current: dict) -> tuple[str, dict[str, float]]:
    reference = baseline["median"]
    thresholds = thresholds_from_baseline(baseline)
    growth = {
        "face_growth": current["face_width"] / reference["face_width"] - 1.0,
        "body_growth": current["shoulder_width"] / reference["shoulder_width"] - 1.0,
        "ratio_growth": current["face_shoulder_ratio"] / reference["face_shoulder_ratio"] - 1.0,
    }

    neck_forward = (
        growth["face_growth"] >= thresholds["face_growth"]
        and growth["ratio_growth"] >= thresholds["ratio_growth"]
    )
    state = "NECK FORWARD" if neck_forward else "NECK NORMAL"
    return state, {**growth, **{f"threshold_{key}": value for key, value in thresholds.items()}}


def data_issue_message(reason: str) -> str:
    if reason == "face_missing":
        return "FACE NOT VISIBLE - MOVE BACK OR SIT UP"
    if reason in ("pose_missing", "pose_world_missing"):
        return "UPPER BODY NOT VISIBLE - SIT UP"
    if reason == "anchor_not_below_chin":
        return "TOO CLOSE OR BENT TOO LOW"
    if reason.endswith("anchor_too_far"):
        return "MOVE BACK OR SIT UP"
    return "CAMERA DATA LOST - CHECK YOUR POSITION"


class SessionLog:
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


def draw_panel(
    frame: np.ndarray,
    state: str,
    current: dict[str, float] | None,
    growth: dict[str, float],
    status: str,
    data_alert: str,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (480, 220), (18, 22, 26), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    state_color = {
        "NECK NORMAL": (80, 220, 150),
        "NECK FORWARD": (70, 110, 245),
    }.get(state, (170, 178, 184))
    cv2.putText(frame, state, (30, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, status, (30, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (165, 174, 182), 1, cv2.LINE_AA)

    rows = (
        ("FACE GROWTH", growth.get("face_growth", 0.0)),
        ("BODY GROWTH", growth.get("body_growth", 0.0)),
        ("RATIO GROWTH", growth.get("ratio_growth", 0.0)),
        ("FACE WIDTH", current.get("face_width", 0.0) if current else 0.0),
        ("VIRTUAL WIDTH", current.get("shoulder_width", 0.0) if current else 0.0),
    )
    for row, (label, value) in enumerate(rows):
        y = 105 + row * 24
        cv2.putText(frame, label, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (150, 160, 168), 1, cv2.LINE_AA)
        text = f"{value * 100:+.1f}%" if "GROWTH" in label else f"{value:.4f}"
        cv2.putText(frame, text, (250, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (232, 234, 236), 1, cv2.LINE_AA)

    cv2.putText(frame, "C calibrate   P pause/release   Y correct   N wrong   Q quit",
                (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (225, 228, 230), 1, cv2.LINE_AA)
    if data_alert:
        overlay = frame.copy()
        cv2.rectangle(overlay, (14, frame.shape[0] - 82),
                      (frame.shape[1] - 14, frame.shape[0] - 42), (45, 55, 210), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.putText(frame, data_alert, (28, frame.shape[0] - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 247, 248), 2, cv2.LINE_AA)


def main() -> None:
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    camera = open_camera()
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera with Media Foundation.")

    log = SessionLog()
    session_started = time.monotonic()
    baseline = None
    rolling: deque[tuple[float, dict[str, float]]] = deque()
    preparing_since: float | None = None
    calibrating_since: float | None = None
    calibration_samples: list[dict[str, float]] = []
    paused = False
    state = "NOT CALIBRATED"
    last_state = state
    status = "Press C and hold your normal comfortable posture"
    current_median = None
    growth: dict[str, float] = {}
    pending_neck_state: str | None = None
    pending_neck_since = 0.0
    data_missing_since: float | None = None
    data_alert_logged = False
    data_alert = ""
    latest_invalid_reason = ""

    with (
        vision.FaceLandmarker.create_from_options(face_options) as face_landmarker,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
    ):
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            while True:
                if paused:
                    frame = np.full((480, 640, 3), (22, 26, 30), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA RELEASED", (185, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 150), 2, cv2.LINE_AA)
                    cv2.putText(frame, "Press P to resume", (222, 258),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (175, 182, 188), 1, cv2.LINE_AA)
                else:
                    ok, frame = camera.read()
                    if not ok:
                        camera.release()
                        paused = True
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int((time.monotonic() - session_started) * 1000)
                    face_result = face_landmarker.detect_for_video(image, timestamp_ms)
                    pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
                    face = face_result.face_landmarks[0] if face_result.face_landmarks else None
                    pose_image = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
                    pose_world = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None
                    metrics, reason = posture_metrics(face, pose_image, pose_world)
                    if face:
                        draw_face(frame, face)

                    now = time.monotonic()
                    if metrics:
                        if data_missing_since is not None and data_alert_logged:
                            log.write(
                                "data_recovered",
                                missing_seconds=now - data_missing_since,
                                previous_reason=latest_invalid_reason,
                            )
                        data_missing_since = None
                        data_alert_logged = False
                        data_alert = ""
                        latest_invalid_reason = ""
                    else:
                        latest_invalid_reason = reason
                        if data_missing_since is None:
                            data_missing_since = now
                        missing_seconds = now - data_missing_since
                        if missing_seconds >= DATA_ALERT_SECONDS:
                            data_alert = data_issue_message(reason)
                            if not data_alert_logged:
                                log.write(
                                    "data_insufficient_alert",
                                    missing_seconds=missing_seconds,
                                    reason=reason,
                                    message=data_alert,
                                )
                                data_alert_logged = True

                    if preparing_since is not None:
                        remaining = PREPARE_SECONDS - (now - preparing_since)
                        if remaining > 0:
                            state = "PREPARE TO CALIBRATE"
                            status = f"Sit normally and hold still: {remaining:.1f}s"
                        else:
                            preparing_since = None
                            calibrating_since = now
                            calibration_samples = []
                            state = "CALIBRATING"

                    if calibrating_since is not None:
                        elapsed = now - calibrating_since
                        if metrics:
                            calibration_samples.append(metrics)
                        status = f"Collecting normal posture: {max(0.0, CALIBRATION_SECONDS - elapsed):.1f}s"
                        if elapsed >= CALIBRATION_SECONDS:
                            baseline, message = build_baseline(calibration_samples)
                            calibrating_since = None
                            rolling.clear()
                            if baseline:
                                state = "CALIBRATED"
                                status = "Building the first 5-second window"
                                log.write("calibrated", baseline=baseline)
                            else:
                                state = "CALIBRATION FAILED"
                                status = message
                                log.write("calibration_failed", reason=message)

                    elif baseline and preparing_since is None:
                        if metrics:
                            rolling.append((now, metrics))
                        while rolling and now - rolling[0][0] > WINDOW_SECONDS:
                            rolling.popleft()
                        if rolling and now - rolling[0][0] >= WINDOW_SECONDS * 0.9:
                            current_median = median_metrics([item[1] for item in rolling])
                            raw_state, growth = classify_posture(baseline, current_median)
                            if state not in ("NECK NORMAL", "NECK FORWARD"):
                                state = raw_state
                                pending_neck_state = None
                            elif raw_state == state:
                                pending_neck_state = None
                            elif pending_neck_state != raw_state:
                                pending_neck_state = raw_state
                                pending_neck_since = now
                            elif now - pending_neck_since >= STATE_CONFIRM_SECONDS:
                                state = raw_state
                                pending_neck_state = None
                            status = (
                                f"Confirming {pending_neck_state.lower()}"
                                if pending_neck_state
                                else "Neck posture only; screen distance ignored"
                            )
                        elif not metrics:
                            state = "DATA INSUFFICIENT"
                            status = reason

                    if state != last_state:
                        log.write("state_changed", previous=last_state, current=state,
                                  growth=growth, metrics=current_median)
                        last_state = state

                    draw_panel(frame, state, current_median, growth, status, data_alert)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c") and not paused:
                    preparing_since = time.monotonic()
                    calibrating_since = None
                    baseline = None
                    rolling.clear()
                    current_median = None
                    growth = {}
                    pending_neck_state = None
                if key == ord("p"):
                    if paused:
                        camera = open_camera()
                        paused = not camera.isOpened()
                        rolling.clear()
                    else:
                        camera.release()
                        paused = True
                        data_missing_since = None
                        data_alert_logged = False
                        data_alert = ""
                    log.write("camera_paused" if paused else "camera_resumed")
                if key in (ord("y"), ord("n")):
                    log.write("user_feedback", state=state,
                              verdict="correct" if key == ord("y") else "wrong",
                              growth=growth, metrics=current_median)
        finally:
            camera.release()
            cv2.destroyAllWindows()
            log.write("session_ended")


if __name__ == "__main__":
    main()
