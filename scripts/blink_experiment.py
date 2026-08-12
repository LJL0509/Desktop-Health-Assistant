import json
import traceback
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from instance_lock import run_with_instance_lock
from landmark_preview import (
    FACE_MODEL,
    LEFT_EYE,
    RIGHT_EYE,
    draw_face,
    eye_open_ratio,
    next_video_timestamp_ms,
    open_camera,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "blink-experiments"
WINDOW_NAME = "Desktop Health Assistant - Blink Experiment"
CALIBRATION_SECONDS = 6.0
MIN_CALIBRATION_SAMPLES = 60
SAMPLE_LOG_INTERVAL_SECONDS = 0.2
LOW_BLINK_WINDOW_SECONDS = 5 * 60
LOW_BLINK_RATE_PER_MINUTE = 15.0
BLINK_RATE_RECOVERY_PER_MINUTE = 18.0
LOW_BLINK_TITLE = "眨眼与眼睛休息提示"
LOW_BLINK_MESSAGE = "最近一段有效观察中的眨眼较少。看看远处，放松眼睛，并自然眨眼。"


def blendshape_scores(result) -> dict[str, float]:
    if not result.face_blendshapes:
        return {}
    categories = result.face_blendshapes[0]
    return {
        category.category_name: float(category.score)
        for category in categories
    }


def eye_metrics(landmarks, result) -> dict[str, float]:
    left = eye_open_ratio(landmarks, LEFT_EYE)
    right = eye_open_ratio(landmarks, RIGHT_EYE)
    scores = blendshape_scores(result)
    return {
        "left_eye_ratio": left,
        "right_eye_ratio": right,
        "left_blink_score": scores.get("eyeBlinkLeft", 0.0),
        "right_blink_score": scores.get("eyeBlinkRight", 0.0),
    }


def build_open_eye_baseline(
    samples: list[dict[str, float]],
    minimum_samples: int = MIN_CALIBRATION_SAMPLES,
) -> tuple[dict | None, str]:
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    if len(samples) < minimum_samples:
        return None, "not enough valid eye samples"
    left = float(np.percentile([item["left_eye_ratio"] for item in samples], 80))
    right = float(np.percentile([item["right_eye_ratio"] for item in samples], 80))
    if min(left, right) < 0.08:
        return None, "eye opening signal is too small"
    asymmetry = abs(left - right) / max(left, right)
    if asymmetry > 0.45:
        return None, "left and right eye signals are too different"
    return {
        "left_open_ratio": left,
        "right_open_ratio": right,
        "left_open_blink_score": float(
            np.median([item["left_blink_score"] for item in samples])
        ),
        "right_open_blink_score": float(
            np.median([item["right_blink_score"] for item in samples])
        ),
        "sample_count": len(samples),
    }, "ok"


def normalized_openness(metrics: dict[str, float], baseline: dict) -> tuple[float, float]:
    return (
        metrics["left_eye_ratio"] / baseline["left_open_ratio"],
        metrics["right_eye_ratio"] / baseline["right_open_ratio"],
    )


@dataclass(frozen=True)
class BlinkEvent:
    kind: str
    occurred_at: float
    duration_seconds: float | None = None


class BlinkDetector:
    def __init__(
        self,
        minimum_closed_seconds: float = 0.05,
        long_closure_seconds: float = 0.8,
        close_ratio: float = 0.45,
        reopen_ratio: float = 0.82,
        close_blendshape_score: float = 0.55,
    ) -> None:
        self.minimum_closed_seconds = minimum_closed_seconds
        self.long_closure_seconds = long_closure_seconds
        self.close_ratio = close_ratio
        self.reopen_ratio = reopen_ratio
        self.close_blendshape_score = close_blendshape_score
        self.state = "OPEN"
        self.closed_since: float | None = None
        self.long_closure_reported = False
        self.high_confidence_closed = False

    def update(
        self,
        now: float,
        left_openness: float | None,
        right_openness: float | None,
        left_blink_score: float | None = None,
        right_blink_score: float | None = None,
    ) -> list[BlinkEvent]:
        if left_openness is None or right_openness is None:
            self.reset()
            return []

        average = (left_openness + right_openness) / 2.0
        has_blendshape_scores = (
            left_blink_score is not None and right_blink_score is not None
        )
        average_blink_score = (
            (left_blink_score + right_blink_score) / 2.0
            if has_blendshape_scores
            else 0.0
        )
        closed = (
            left_openness <= 0.70
            and right_openness <= 0.70
            and (
                average <= self.close_ratio
                or average_blink_score >= self.close_blendshape_score
            )
        )
        high_confidence_closed = (
            left_openness <= 0.50
            and right_openness <= 0.50
            and average <= 0.35
            and left_blink_score is not None
            and right_blink_score is not None
            and left_blink_score >= 0.45
            and right_blink_score >= 0.45
            and average_blink_score >= 0.55
        )
        geometrically_reopened = (
            left_openness >= 0.72
            and right_openness >= 0.72
            and average >= self.reopen_ratio
        )
        blendshape_reopened = (
            has_blendshape_scores
            and left_openness >= 0.50
            and right_openness >= 0.50
            and average >= 0.55
            and average_blink_score <= 0.40
        )
        reopened = geometrically_reopened or blendshape_reopened
        events = []

        if self.state == "OPEN":
            if closed:
                self.state = "CLOSING"
                self.closed_since = now
                self.high_confidence_closed = high_confidence_closed
            return events

        if self.closed_since is None:
            self.reset()
            return events

        duration = max(0.0, now - self.closed_since)
        if self.state == "CLOSING":
            if reopened:
                if self.high_confidence_closed:
                    events.append(BlinkEvent("blink", now, duration))
                self.reset()
            elif closed and duration >= self.minimum_closed_seconds:
                self.state = "CLOSED"
            return events

        if not self.long_closure_reported and duration >= self.long_closure_seconds:
            self.long_closure_reported = True
            events.append(BlinkEvent("long_eye_closure_started", now, duration))

        if reopened:
            if self.long_closure_reported:
                events.append(BlinkEvent("long_eye_closure_ended", now, duration))
            else:
                events.append(BlinkEvent("blink", now, duration))
            self.reset()
        return events

    def reset(self) -> None:
        self.state = "OPEN"
        self.closed_since = None
        self.long_closure_reported = False
        self.high_confidence_closed = False


class BlinkRateMonitor:
    def __init__(
        self,
        window_seconds: float = LOW_BLINK_WINDOW_SECONDS,
        low_rate_per_minute: float = LOW_BLINK_RATE_PER_MINUTE,
        recovery_rate_per_minute: float = BLINK_RATE_RECOVERY_PER_MINUTE,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 <= low_rate_per_minute < recovery_rate_per_minute:
            raise ValueError("recovery rate must be greater than low rate")
        self.window_seconds = window_seconds
        self.low_rate_per_minute = low_rate_per_minute
        self.recovery_rate_per_minute = recovery_rate_per_minute
        self.effective_time = 0.0
        self.last_updated_at: float | None = None
        self.blink_times: deque[float] = deque()
        self.alerted = False
        self.alert_count = 0

    def update(self, now: float, valid: bool, blink: bool = False) -> list[dict]:
        if self.last_updated_at is not None:
            elapsed = max(0.0, now - self.last_updated_at)
            if valid:
                self.effective_time += min(elapsed, 1.0)
        self.last_updated_at = now
        if blink and valid:
            self.blink_times.append(self.effective_time)

        cutoff = self.effective_time - self.window_seconds
        while self.blink_times and self.blink_times[0] <= cutoff:
            self.blink_times.popleft()
        if self.effective_time < self.window_seconds:
            return []

        rate = self.rate_per_minute()
        if not self.alerted and rate < self.low_rate_per_minute:
            self.alerted = True
            self.alert_count += 1
            return [{
                "event": "low_blink_rate_alert",
                "rate_per_minute": rate,
                "blink_count": len(self.blink_times),
                "valid_observation_seconds": self.window_seconds,
                "threshold_per_minute": self.low_rate_per_minute,
            }]
        if self.alerted and rate >= self.recovery_rate_per_minute:
            self.alerted = False
            return [{
                "event": "low_blink_rate_recovered",
                "rate_per_minute": rate,
                "blink_count": len(self.blink_times),
                "valid_observation_seconds": self.window_seconds,
            }]
        return []

    def rate_per_minute(self) -> float | None:
        denominator = min(self.effective_time, self.window_seconds)
        if denominator <= 0:
            return None
        return len(self.blink_times) * 60.0 / denominator


class BlinkNotifier:
    def __init__(self) -> None:
        from windows_toasts import InteractableWindowsToaster

        self.toaster = InteractableWindowsToaster("Desktop Health Assistant")

    def show_low_blink_rate(self) -> None:
        from windows_toasts import Toast, ToastAudio, ToastDuration, ToastScenario

        self.toaster.show_toast(
            Toast(
                text_fields=[LOW_BLINK_TITLE, LOW_BLINK_MESSAGE],
                audio=ToastAudio(silent=True),
                duration=ToastDuration.Long,
                scenario=ToastScenario.Reminder,
            )
        )


class ExperimentLog:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = DATA_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"

    def write(self, event: str, **details) -> None:
        payload = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
            **details,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def draw_panel(
    frame: np.ndarray,
    metrics: dict[str, float] | None,
    baseline: dict | None,
    detector: BlinkDetector,
    blink_count: int,
    blinks_last_minute: int,
    status: str,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (555, 262), (18, 22, 26), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    if metrics and baseline:
        left_open, right_open = normalized_openness(metrics, baseline)
    else:
        left_open = right_open = 0.0
    lines = (
        ("STATUS", status),
        ("EYE STATE", detector.state),
        ("BLINKS", str(blink_count)),
        ("LAST 60 S", str(blinks_last_minute)),
        (
            "EYE RATIO L/R",
            "N/A" if not metrics else f"{metrics['left_eye_ratio']:.3f} / {metrics['right_eye_ratio']:.3f}",
        ),
        (
            "NORMALIZED L/R",
            "N/A" if not baseline or not metrics else f"{left_open:.2f} / {right_open:.2f}",
        ),
        (
            "BLENDSHAPE L/R",
            "N/A" if not metrics else f"{metrics['left_blink_score']:.2f} / {metrics['right_blink_score']:.2f}",
        ),
    )
    for row, (label, value) in enumerate(lines):
        y = 40 + row * 29
        cv2.putText(frame, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (155, 164, 172), 1, cv2.LINE_AA)
        color = (80, 220, 150) if detector.state == "OPEN" else (70, 180, 245)
        cv2.putText(frame, value, (184, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
                    color, 1, cv2.LINE_AA)

    cv2.putText(frame, "C calibrate   R reset count   Q quit", (22, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 232, 234), 1, cv2.LINE_AA)


def main() -> None:
    if not FACE_MODEL.exists():
        raise FileNotFoundError("Missing face_landmarker.task; run scripts/download_models.py")

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    log = ExperimentLog()
    log.write("experiment_started")
    detector = BlinkDetector()
    baseline = None
    calibration_started = None
    calibration_samples = []
    blink_times: deque[float] = deque()
    blink_count = 0
    last_sample_log = 0.0
    started = time.monotonic()
    video_timestamp_ms = -1
    status = "Press C and look at the screen normally"
    exit_reason = "unknown"
    caught_error = None

    print("Loading face model...", flush=True)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        camera = open_camera()
        if not camera.isOpened():
            raise RuntimeError("Could not open the camera with Media Foundation.")
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            while True:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = "window_closed"
                    log.write("exit_requested", reason=exit_reason)
                    break
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame could not be read.")
                frame = cv2.flip(frame, 1)
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now = time.monotonic()
                video_timestamp_ms = next_video_timestamp_ms(
                    now - started,
                    video_timestamp_ms,
                )
                result = landmarker.detect_for_video(image, video_timestamp_ms)
                face = result.face_landmarks[0] if result.face_landmarks else None
                metrics = eye_metrics(face, result) if face is not None else None
                if face is not None:
                    draw_face(frame, face)

                if calibration_started is not None:
                    elapsed = now - calibration_started
                    if metrics:
                        calibration_samples.append(metrics)
                    status = f"Calibrating open eyes: {max(0.0, CALIBRATION_SECONDS - elapsed):.1f}s"
                    if elapsed >= CALIBRATION_SECONDS:
                        baseline, message = build_open_eye_baseline(calibration_samples)
                        if baseline:
                            status = "Calibrated - blink naturally"
                            log.write("calibration_completed", baseline=baseline)
                        else:
                            status = f"Calibration failed: {message}"
                            log.write("calibration_failed", reason=message)
                        calibration_started = None
                        calibration_samples = []
                        detector.reset()

                if baseline and calibration_started is None:
                    if metrics:
                        left_open, right_open = normalized_openness(metrics, baseline)
                        events = detector.update(
                            now,
                            left_open,
                            right_open,
                            metrics["left_blink_score"],
                            metrics["right_blink_score"],
                        )
                    else:
                        left_open = right_open = None
                        events = detector.update(now, None, None)
                        status = "Face missing - blink counting paused"

                    for event in events:
                        if event.kind == "blink":
                            blink_count += 1
                            blink_times.append(now)
                            status = "Blink detected"
                        elif event.kind == "long_eye_closure_started":
                            status = "Eyes closed - not counted as a blink yet"
                        elif event.kind == "long_eye_closure_ended":
                            status = "Long eye closure ended"
                        log.write(
                            event.kind,
                            duration_seconds=event.duration_seconds,
                            left_openness=left_open,
                            right_openness=right_open,
                            metrics=metrics,
                        )

                    if metrics and now - last_sample_log >= SAMPLE_LOG_INTERVAL_SECONDS:
                        log.write(
                            "eye_sample",
                            detector_state=detector.state,
                            left_openness=left_open,
                            right_openness=right_open,
                            metrics=metrics,
                        )
                        last_sample_log = now

                while blink_times and now - blink_times[0] > 60.0:
                    blink_times.popleft()
                draw_panel(
                    frame,
                    metrics,
                    baseline,
                    detector,
                    blink_count,
                    len(blink_times),
                    status,
                )
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    exit_reason = "keyboard_q" if key == ord("q") else "keyboard_escape"
                    log.write("exit_requested", reason=exit_reason, key_code=key)
                    break
                if key == ord("c"):
                    calibration_started = now
                    calibration_samples = []
                    baseline = None
                    detector.reset()
                    status = "Look at the screen normally; brief natural blinks are okay"
                    log.write("calibration_started")
                if key == ord("r"):
                    blink_count = 0
                    blink_times.clear()
                    detector.reset()
                    log.write("blink_count_reset")
        except Exception as error:
            exit_reason = "exception"
            caught_error = error
            log.write(
                "experiment_error",
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        finally:
            camera.release()
            cv2.destroyAllWindows()
            log.write(
                "experiment_ended",
                blink_count=blink_count,
                calibrated=baseline is not None,
                reason=exit_reason,
            )
            print(f"Numerical log: {log.path}")
    if caught_error is not None:
        raise caught_error


if __name__ == "__main__":
    raise SystemExit(run_with_instance_lock(main))
