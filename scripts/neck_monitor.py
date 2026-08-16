import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image, ImageDraw, ImageFont

from activity_monitor import ActivityMonitorService, ContinuousUseTracker
from app_paths import app_version, data_path
from blink_experiment import (
    BlinkDetector,
    BlinkNotifier,
    BlinkRateMonitor,
    build_open_eye_baseline,
    eye_metrics,
    normalized_openness,
)
from daily_report import AutoReportScheduler
from health_popup import (
    FIRST_REPEAT_REMINDER_SECONDS,
    HealthPopupNotifier,
    ONGOING_REPEAT_REMINDER_SECONDS,
)
from instance_lock import run_with_instance_lock
from landmark_preview import draw_face, next_video_timestamp_ms, open_camera
from reminder_service import (
    DEFAULT_WATER_ML,
    ReminderScheduler,
    ReminderService,
    ReminderStore,
)
from tray_control import (
    format_elapsed_time,
    hydration_elapsed_seconds,
    set_native_window_visible,
)
from upper_body_contour_experiment import (
    FACE_MODEL,
    SEGMENTER_MODEL,
    classify_motion,
    stabilized_visibility_mode,
    upper_body_metrics,
)


APP_VERSION = app_version()
DATA_DIR = data_path("monitor-sessions")
WINDOW_NAME = f"Desktop Health Assistant - Neck Monitor v{APP_VERSION}"
CALIBRATION_SECONDS = 10.0
MAX_CALIBRATION_SECONDS = 20.0
MIN_INTEGRATED_CALIBRATION_SAMPLES = 30
CURRENT_WINDOW_SECONDS = 2.0
PREPARE_SECONDS = 3.0
STATE_CONFIRM_SECONDS = 2.0
DATA_ALERT_SECONDS = 10.0
QUALITY_WINDOW_SIZE = 30
POSTURE_ALERT_SECONDS = 60.0
HEAD_TOO_LOW_ALERT_SECONDS = 60.0
ISSUE_RECOVERY_GRACE_SECONDS = 3.0
REMINDER_POLL_SECONDS = 1.0
UI_FONT = Path("C:/Windows/Fonts/msyh.ttc")
UI_FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")
HYDRATION_PANEL_RECT = (366, 350, 626, 466)
WATER_BUTTON_RECT = (470, 408, 610, 451)
DATA_ALERT_RECT = (14, 342, 354, 466)
UI_REFERENCE_SIZE = (640, 480)
ISSUE_MESSAGES = {
    "neck_forward": "HEAD FORWARD FOR TOO LONG - RETURN TO A COMFORTABLE POSTURE",
    "head_too_low": "HEAD TOO LOW FOR TOO LONG - RAISE IT SLIGHTLY",
}
POSTURE_NOTIFICATION_TEXT = {
    "neck_forward": (
        "姿势提醒",
        "头部前倾已经持续一段时间。请让头部回到躯干上方，活动一下再继续。",
    ),
    "head_too_low": (
        "姿势提醒",
        "低头姿势已经持续一段时间。请稍微抬高头部，并调整屏幕或阅读位置。",
    ),
}


@lru_cache(maxsize=12)
def ui_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = UI_FONT_BOLD if bold else UI_FONT
    return ImageFont.truetype(str(path), size)


@lru_cache(maxsize=512)
def text_bitmap(
    text: str,
    size: int,
    bold: bool,
    color: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    font = ui_font(size, bold)
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)))
    ImageDraw.Draw(image).text((-left, -top), text, font=font, fill=color)
    return np.asarray(image), (left, top)


def draw_ui_text(
    frame: np.ndarray,
    position: tuple[int, int],
    text: str,
    size: int,
    color: tuple[int, int, int, int],
    bold: bool = False,
) -> None:
    bitmap, offset = text_bitmap(text, size, bold, color)
    x = position[0] + offset[0]
    y = position[1] + offset[1]
    height, width = bitmap.shape[:2]
    frame_height, frame_width = frame.shape[:2]
    source_left = max(0, -x)
    source_top = max(0, -y)
    source_right = min(width, frame_width - x)
    source_bottom = min(height, frame_height - y)
    if source_left >= source_right or source_top >= source_bottom:
        return

    target = frame[
        y + source_top : y + source_bottom,
        x + source_left : x + source_right,
    ]
    source = bitmap[source_top:source_bottom, source_left:source_right]
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    source_bgr = source[:, :, :3][:, :, ::-1].astype(np.float32)
    target[:] = (source_bgr * alpha + target * (1.0 - alpha)).astype(np.uint8)


@lru_cache(maxsize=8)
def solid_color_frame(
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = color[::-1]
    return frame


@lru_cache(maxsize=16)
def rounded_corner_restore_masks(radius: int) -> tuple[np.ndarray, ...]:
    y, x = np.ogrid[:radius, :radius]
    top_left = np.where(
        (x - radius) ** 2 + (y - radius) ** 2 > radius**2,
        255,
        0,
    ).astype(np.uint8)
    return (
        top_left,
        np.fliplr(top_left),
        np.flipud(top_left),
        np.flip(top_left),
    )


def blend_rounded_rectangle(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    radius: int,
    color: tuple[int, int, int],
    opacity: float,
) -> None:
    left, top, right, bottom = box
    left = max(0, left)
    top = max(0, top)
    right = min(frame.shape[1], right)
    bottom = min(frame.shape[0], bottom)
    if left >= right or top >= bottom:
        return

    target = frame[top:bottom, left:right]
    height, width = target.shape[:2]
    radius = max(0, min(radius, width // 2, height // 2))
    corners = ()
    if radius:
        corners = (
            target[:radius, :radius].copy(),
            target[:radius, -radius:].copy(),
            target[-radius:, :radius].copy(),
            target[-radius:, -radius:].copy(),
        )

    cv2.addWeighted(
        solid_color_frame(width, height, color),
        opacity,
        target,
        1.0 - opacity,
        0,
        dst=target,
    )
    if radius:
        targets = (
            target[:radius, :radius],
            target[:radius, -radius:],
            target[-radius:, :radius],
            target[-radius:, -radius:],
        )
        for original, mask, corner in zip(
            corners,
            rounded_corner_restore_masks(radius),
            targets,
        ):
            cv2.copyTo(original, mask, corner)


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def scaled_ui_rect(
    rect: tuple[int, int, int, int],
    frame_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = frame_size
    scale_x = width / UI_REFERENCE_SIZE[0]
    scale_y = height / UI_REFERENCE_SIZE[1]
    left, top, right, bottom = rect
    return (
        round(left * scale_x),
        round(top * scale_y),
        round(right * scale_x),
        round(bottom * scale_y),
    )


def display_size_from_window_rect(
    window_rect: tuple[int, int, int, int],
    fallback: tuple[int, int] = UI_REFERENCE_SIZE,
) -> tuple[int, int]:
    width, height = window_rect[2:]
    if width <= 0 or height <= 0:
        return fallback
    return width, height


def current_display_size(
    window_name: str,
    fallback: tuple[int, int] = UI_REFERENCE_SIZE,
) -> tuple[int, int]:
    try:
        return display_size_from_window_rect(
            cv2.getWindowImageRect(window_name),
            fallback,
        )
    except cv2.error:
        return fallback


def format_reminder_countdown(seconds: float) -> str:
    remaining = max(0, int(np.ceil(seconds)))
    minutes, seconds = divmod(remaining, 60)
    return f"{minutes}m {seconds:02d}s"


def wrap_ui_text(text: str, size: int, max_width: int) -> list[str]:
    font = ui_font(size)
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


class PostureNotifier:
    def __init__(self, notifier: HealthPopupNotifier | None = None) -> None:
        self.notifier = notifier or HealthPopupNotifier()

    def show(self, issue: str) -> None:
        title, message = POSTURE_NOTIFICATION_TEXT[issue]
        self.notifier.show(title, message)


class LatestFrameWorker:
    def __init__(self, processor_factory: Callable[[], Callable]) -> None:
        self.processor_factory = processor_factory
        self.condition = threading.Condition()
        self.pending = None
        self.latest_sequence = -1
        self.latest_value = None
        self.error: BaseException | None = None
        self.stopping = False
        self.next_sequence = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, payload) -> int:
        with self.condition:
            if self.stopping:
                raise RuntimeError("Cannot submit to a stopped worker")
            sequence = self.next_sequence
            self.next_sequence += 1
            self.pending = (sequence, payload)
            self.condition.notify()
            return sequence

    def latest(self) -> tuple[int, object | None]:
        with self.condition:
            error = self.error
            sequence = self.latest_sequence
            value = self.latest_value
        if error is not None:
            raise RuntimeError("Background segmentation failed") from error
        return sequence, value

    def close(self) -> None:
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join()

    def _run(self) -> None:
        processor = None
        try:
            processor = self.processor_factory()
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.stopping or self.pending is not None
                    )
                    if self.stopping:
                        return
                    sequence, payload = self.pending
                    self.pending = None
                value = processor(payload)
                with self.condition:
                    self.latest_sequence = sequence
                    self.latest_value = value
        except BaseException as error:
            with self.condition:
                self.error = error
        finally:
            close = getattr(processor, "close", None)
            if close is not None:
                close()


class SegmentationProcessor:
    def __init__(self) -> None:
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=False,
            output_category_mask=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def __call__(self, payload: tuple[np.ndarray, int]) -> np.ndarray:
        rgb, timestamp_ms = payload
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.segmenter.segment_for_video(image, timestamp_ms)
        return np.array(result.category_mask.numpy_view(), copy=True)

    def close(self) -> None:
        self.segmenter.close()


def median_metrics(samples: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in samples[0] if key != "elapsed_seconds"]
    return {
        key: float(np.median([sample[key] for sample in samples]))
        for key in keys
    }


def mad_metrics(
    samples: list[dict[str, float]],
    medians: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(np.median([abs(sample[key] - medians[key]) for sample in samples]))
        for key in medians
    }


def baseline_from_samples(samples: list[dict[str, float]]) -> dict:
    medians = median_metrics(samples)
    mads = mad_metrics(samples, medians)
    stability = {
        key: mads[key] / max(abs(medians[key]), 1e-6)
        for key in ("face_width", "torso_width", "face_torso_ratio")
    }
    return {
        "median": medians,
        "mad": mads,
        "relative_mad": stability,
    }


def build_baseline(samples: list[dict[str, float]]) -> tuple[dict | None, str]:
    if len(samples) < 30:
        return None, "not enough valid contour data"
    baseline = baseline_from_samples(samples)
    stability = baseline["relative_mad"]
    if stability["face_width"] > 0.06:
        return None, "face scale was unstable"
    if stability["torso_width"] > 0.10:
        return None, "upper body contour was unstable"
    if stability["face_torso_ratio"] > 0.10:
        return None, "face/body ratio was unstable"
    return baseline, "ok"


def relative_features(reference: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    return {
        "face_growth": current["face_width"] / reference["face_width"] - 1.0,
        "torso_growth": current["torso_width"] / reference["torso_width"] - 1.0,
        "face_y_change": current["face_center_y"] - reference["face_center_y"],
        "torso_y_change": current["torso_center_y"] - reference["torso_center_y"],
        "torso_area_growth": current["torso_area"] / reference["torso_area"] - 1.0,
        "ratio_growth": (
            current["face_torso_ratio"] / reference["face_torso_ratio"] - 1.0
        ),
    }


def classify_relative_motion(
    features: dict[str, float],
    partial_view: bool = False,
) -> str:
    face_growth = features["face_growth"]
    torso_growth = features["torso_growth"]
    ratio_growth = features["ratio_growth"]
    face_y_change = features["face_y_change"]
    torso_y_change = features["torso_y_change"]
    torso_area_growth = features["torso_area_growth"]
    scale_difference = face_growth - torso_growth
    vertical_difference = face_y_change - torso_y_change

    coordinated_back = (
        face_growth <= -0.06
        and torso_growth <= -0.06
        and ratio_growth <= 0.08
    )
    if coordinated_back:
        return "WHOLE BODY BACK"

    motion = classify_motion(features)
    partial_head_forward = motion == "HEAD FORWARD" and torso_growth > -0.30
    if partial_head_forward:
        return "HEAD FORWARD"

    severe_contour_loss = torso_area_growth <= -0.60 or torso_growth <= -0.35
    if severe_contour_loss and not partial_view:
        return "DATA INSUFFICIENT"
    if severe_contour_loss:
        return "STABLE"

    if motion == "HEAD FORWARD":
        return motion

    coordinated_forward = (
        face_growth >= 0.08
        and torso_growth >= 0.07
        and abs(ratio_growth) <= 0.10
    )
    if coordinated_forward:
        return "WHOLE BODY FORWARD"

    relative_head_forward = ratio_growth >= 0.09 and scale_difference >= 0.06
    mild_head_forward = (
        face_growth >= 0.05
        and torso_growth <= 0.06
        and ratio_growth >= 0.05
        and scale_difference >= 0.05
        and (face_y_change >= 0.015 or torso_area_growth <= -0.12)
    )
    lowered_head_forward = (
        ratio_growth >= 0.07
        and vertical_difference >= 0.03
        and torso_area_growth <= -0.20
    )
    if relative_head_forward or mild_head_forward or lowered_head_forward:
        return "HEAD FORWARD"

    if motion != "UNCERTAIN":
        return motion

    normal_jitter = (
        abs(face_growth) <= 0.06
        and abs(torso_growth) <= 0.09
        and abs(ratio_growth) <= 0.08
        and abs(vertical_difference) <= 0.04
    )
    if normal_jitter:
        return "STABLE"

    clear_non_forward = (
        face_growth <= 0.05
        or ratio_growth <= 0.05
        or scale_difference <= 0.05
        or (
            vertical_difference < 0.015
            and torso_area_growth > -0.12
        )
    )
    if clear_non_forward:
        return "STABLE"

    return motion


def target_posture_state(
    current_state: str,
    motion: str,
    features: dict[str, float],
) -> str:
    if motion == "HEAD FORWARD":
        return "NECK FORWARD"
    if motion in ("STABLE", "WHOLE BODY FORWARD", "WHOLE BODY BACK"):
        return "NECK NORMAL"
    return current_state


def assess_posture(
    baseline: dict,
    current: dict[str, float],
    current_state: str = "NECK NORMAL",
    partial_view: bool = False,
) -> tuple[str, dict[str, float], str]:
    features = relative_features(baseline["median"], current)
    motion = classify_relative_motion(features, partial_view=partial_view)
    state = target_posture_state(current_state, motion, features)
    return state, features, motion


def classify_posture(baseline: dict, current: dict) -> tuple[str, dict[str, float]]:
    state, features, _ = assess_posture(baseline, current)
    return state, features


def data_issue_message(reason: str) -> str:
    if reason == "head_too_low":
        return "HEAD TOO LOW - RAISE IT SLIGHTLY"
    if reason == "face_missing":
        return "FACE NOT VISIBLE - MOVE BACK OR SIT UP"
    if reason == "contour_unstable":
        return "UPPER BODY DATA LOST - CHECK YOUR POSITION"
    return "CAMERA DATA LOST - CHECK YOUR POSITION"


def resolve_visibility_mode(
    mode: str,
    last_reliable_mode: str | None,
    metrics_available: bool,
    face_available: bool,
    clearance_ratio: float,
) -> str:
    if mode != "CONTOUR UNSTABLE" or not face_available:
        return mode
    if metrics_available:
        low_threshold = 0.18 if last_reliable_mode == "TOO LOW" else 0.15
        if clearance_ratio < low_threshold:
            return "TOO LOW"
        if last_reliable_mode in ("PARTIAL", "TOO LOW"):
            return "PARTIAL"
    if last_reliable_mode == "TOO LOW":
        return "TOO LOW"
    return mode


class IssueAccumulator:
    def __init__(
        self,
        alert_seconds: float | None = None,
        recovery_grace_seconds: float = ISSUE_RECOVERY_GRACE_SECONDS,
        first_repeat_seconds: float = FIRST_REPEAT_REMINDER_SECONDS,
        repeat_seconds: float = ONGOING_REPEAT_REMINDER_SECONDS,
    ) -> None:
        self.alert_seconds_by_issue = {
            "neck_forward": (
                POSTURE_ALERT_SECONDS if alert_seconds is None else alert_seconds
            ),
            "head_too_low": (
                HEAD_TOO_LOW_ALERT_SECONDS if alert_seconds is None else alert_seconds
            ),
        }
        self.recovery_grace_seconds = recovery_grace_seconds
        self.first_repeat_seconds = first_repeat_seconds
        self.repeat_seconds = repeat_seconds
        self.current_issue: str | None = None
        self.started_at = 0.0
        self.current_issue_started_at = 0.0
        self.last_seen_at = 0.0
        self.alerted = False
        self.current_issue_alerted = False
        self.next_alert_at: float | None = None
        self.statistics = {
            issue: {
                "total_seconds": 0.0,
                "episode_count": 0,
                "longest_seconds": 0.0,
                "alert_count": 0,
            }
            for issue in ISSUE_MESSAGES
        }

    def alert_threshold(self, issue: str | None = None) -> float:
        selected = issue or self.current_issue
        if selected is None:
            return POSTURE_ALERT_SECONDS
        return self.alert_seconds_by_issue[selected]

    def _begin_issue(self, now: float, issue: str) -> dict:
        self.current_issue = issue
        self.current_issue_started_at = now
        self.last_seen_at = now
        self.current_issue_alerted = False
        return {"event": "posture_issue_started", "issue": issue}

    def _start(self, now: float, issue: str) -> dict:
        self.started_at = now
        self.alerted = False
        self.next_alert_at = None
        return self._begin_issue(now, issue)

    def _end_current_issue(self, ended_at: float) -> dict:
        issue = self.current_issue
        if issue is None:
            raise RuntimeError("Cannot close a posture issue when none is active")
        duration = max(0.0, ended_at - self.current_issue_started_at)
        stats = self.statistics[issue]
        stats["total_seconds"] += duration
        stats["episode_count"] += 1
        stats["longest_seconds"] = max(stats["longest_seconds"], duration)
        event = {
            "event": "posture_issue_ended",
            "issue": issue,
            "duration_seconds": duration,
            "alerted": self.current_issue_alerted,
        }
        self.current_issue = None
        self.current_issue_started_at = 0.0
        self.last_seen_at = 0.0
        self.current_issue_alerted = False
        return event

    def _close(self, ended_at: float) -> dict:
        event = self._end_current_issue(ended_at)
        self.started_at = 0.0
        self.alerted = False
        self.next_alert_at = None
        return event

    def _alert_event(self, now: float, duration: float, repeat: bool) -> dict:
        issue = self.current_issue
        if issue is None:
            raise RuntimeError("Cannot alert when no posture issue is active")
        self.alerted = True
        self.current_issue_alerted = True
        self.statistics[issue]["alert_count"] += 1
        self.next_alert_at = now + (
            self.repeat_seconds if repeat else self.first_repeat_seconds
        )
        return {
            "event": "posture_alert",
            "issue": issue,
            "duration_seconds": duration,
            "threshold_seconds": self.alert_threshold(issue),
            "repeat": repeat,
        }

    def update(self, now: float, issue: str | None) -> list[dict]:
        events: list[dict] = []
        if self.current_issue is None:
            if issue is not None:
                events.append(self._start(now, issue))
            return events

        if issue == self.current_issue:
            self.last_seen_at = now
        elif issue is not None:
            events.append(self._end_current_issue(now))
            events.append(self._begin_issue(now, issue))
        elif now - self.last_seen_at >= self.recovery_grace_seconds:
            events.append(self._close(self.last_seen_at))
            return events

        duration = self.last_seen_at - self.started_at
        if not self.alerted and duration >= self.alert_threshold():
            events.append(self._alert_event(self.last_seen_at, duration, False))
        elif (
            self.alerted
            and self.next_alert_at is not None
            and self.last_seen_at >= self.next_alert_at
        ):
            events.append(self._alert_event(self.last_seen_at, duration, True))
        return events

    def finish(self, now: float) -> list[dict]:
        if self.current_issue is None:
            return []
        ended_at = min(now, self.last_seen_at)
        return [self._close(ended_at)]

    def active_duration(self, now: float) -> float:
        if self.current_issue is None:
            return 0.0
        return max(0.0, min(now, self.last_seen_at) - self.started_at)

    def seconds_until_next_alert(self, now: float) -> float | None:
        if not self.alerted or self.next_alert_at is None:
            return None
        return max(0.0, self.next_alert_at - now)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            issue: dict(statistics)
            for issue, statistics in self.statistics.items()
        }


class SessionLog:
    def __init__(self, started_at: datetime | None = None) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = (started_at or datetime.now()).strftime("%Y%m%d-%H%M%S")
        self.path = DATA_DIR / f"{timestamp}.jsonl"

    def write(
        self,
        event: str,
        occurred_at: datetime | None = None,
        **details,
    ) -> None:
        payload = {
            "timestamp": (occurred_at or datetime.now().astimezone()).isoformat(
                timespec="seconds"
            ),
            "event": event,
            **details,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def draw_panel(
    frame: np.ndarray,
    state: str,
    current: dict[str, float] | None,
    features: dict[str, float],
    motion: str,
    mode: str,
    status: str,
    data_alert: str,
    blink_count: int,
    blink_rate: float | None,
    hydration_elapsed: float | None = None,
    hydration_overdue: bool = False,
) -> None:
    height, width = frame.shape[:2]
    scale_x = width / UI_REFERENCE_SIZE[0]
    scale_y = height / UI_REFERENCE_SIZE[1]
    font_scale = min(scale_x, scale_y)

    def xy(x: int, y: int) -> tuple[int, int]:
        return round(x * scale_x), round(y * scale_y)

    def rect(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return scaled_ui_rect(box, (width, height))

    def radius(value: int) -> int:
        return max(1, round(value * font_scale))

    def font_size(size: int) -> int:
        return max(8, round(size * font_scale))

    regular = font_size(13)
    small = font_size(11)
    value_font = font_size(12)
    title_font = font_size(22)
    button_font = font_size(12)

    blend_rounded_rectangle(
        frame,
        rect((14, 14, 354, 330)),
        radius(7),
        (18, 22, 26),
        224 / 255,
    )
    blend_rounded_rectangle(
        frame,
        rect(HYDRATION_PANEL_RECT),
        radius(7),
        (18, 22, 26),
        232 / 255,
    )

    state_color = {
        "NECK NORMAL": (80, 220, 150, 255),
        "NECK FORWARD": (245, 110, 70, 255),
        "HEAD TOO LOW": (245, 110, 70, 255),
    }.get(state, (184, 190, 196, 255))
    draw_ui_text(frame, xy(29, 28), state, title_font, state_color, bold=True)
    compact_status = status if len(status) <= 44 else status[:41] + "..."
    draw_ui_text(
        frame,
        xy(29, 61),
        compact_status,
        small,
        (174, 182, 189, 255),
    )

    rows = (
        ("MOTION", motion),
        ("VISIBILITY", mode),
        ("FACE GROWTH", features.get("face_growth", 0.0)),
        ("TORSO GROWTH", features.get("torso_growth", 0.0)),
        ("RATIO GROWTH", features.get("ratio_growth", 0.0)),
        ("HEAD CLEARANCE", current.get("head_clearance_ratio", 0.0) if current else 0.0),
        ("FACE / TORSO", current.get("face_torso_ratio", 0.0) if current else 0.0),
        ("BLINKS", str(blink_count)),
        ("BLINK RATE", "N/A" if blink_rate is None else f"{blink_rate:.1f} / min"),
    )
    for row, (label, value) in enumerate(rows):
        y = 91 + row * 25
        draw_ui_text(frame, xy(29, y), label, small, (145, 155, 163, 255))
        if isinstance(value, str):
            text = value
        elif "GROWTH" in label:
            text = f"{value * 100:+.1f}%"
        else:
            text = f"{value:.4f}"
        draw_ui_text(
            frame,
            xy(215, y),
            text,
            value_font,
            (234, 237, 239, 255),
        )

    hydration_color = (
        (245, 166, 72, 255) if hydration_overdue else (88, 205, 157, 255)
    )
    draw_ui_text(
        frame,
        xy(382, 364),
        "补水与休息",
        regular,
        (226, 231, 234, 255),
    )
    draw_ui_text(
        frame,
        xy(382, 390),
        format_elapsed_time(hydration_elapsed),
        button_font,
        hydration_color,
        bold=True,
    )
    blend_rounded_rectangle(
        frame,
        rect(WATER_BUTTON_RECT),
        radius(5),
        (200, 68, 85),
        1.0,
    )
    draw_ui_text(
        frame,
        xy(489, 419),
        "已补水 250 ml",
        button_font,
        (255, 255, 255, 255),
        bold=True,
    )

    if data_alert:
        blend_rounded_rectangle(
            frame,
            rect(DATA_ALERT_RECT),
            radius(5),
            (210, 55, 45),
            232 / 255,
        )
        lines = wrap_ui_text(data_alert, regular, round(312 * scale_x))
        for row, line in enumerate(lines[:4]):
            draw_ui_text(
                frame,
                xy(28, 352 + row * 24),
                line,
                regular,
                (248, 249, 250, 255),
            )


def draw_paused_message(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    scale = min(width / UI_REFERENCE_SIZE[0], height / UI_REFERENCE_SIZE[1])
    cv2.putText(
        frame,
        "CAMERA RELEASED",
        (round(width * 0.29), round(height * 0.46)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9 * scale,
        (80, 220, 150),
        max(2, round(2 * scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press P to resume",
        (round(width * 0.35), round(height * 0.54)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * scale,
        (175, 182, 188),
        max(1, round(scale)),
        cv2.LINE_AA,
    )


def main(control=None) -> None:
    missing = [path for path in (FACE_MODEL, SEGMENTER_MODEL) if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing {names}; run scripts/download_models.py")

    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    log = SessionLog()
    log.write("session_started", monitor_version=APP_VERSION)
    session_started = time.monotonic()
    video_started = session_started
    video_timestamp_ms = -1
    baseline = None
    rolling: deque[tuple[float, dict[str, float]]] = deque()
    quality_history: deque[tuple[float, bool]] = deque(maxlen=QUALITY_WINDOW_SIZE)
    preparing_since: float | None = None
    calibrating_since: float | None = None
    calibration_samples: list[dict[str, float]] = []
    eye_calibration_samples: list[dict[str, float]] = []
    blink_baseline = None
    blink_detector = BlinkDetector()
    blink_rate_monitor = BlinkRateMonitor()
    blink_notifier = BlinkNotifier()
    posture_notifier = PostureNotifier()
    blink_count = 0
    long_closure_count = 0
    long_closure_seconds = 0.0
    eye_sample_count = 0
    completed_eye_observation_seconds = 0.0
    category_mask_cache = None
    segmentation_result_sequence = -1
    paused = False
    window_visible = True
    posture_state = "NECK NORMAL"
    display_state = "NOT CALIBRATED"
    last_display_state = display_state
    status = "Press C and hold your normal comfortable posture"
    current_median = None
    features: dict[str, float] = {}
    motion = "WAITING"
    pending_state: str | None = None
    pending_since = 0.0
    mode = "CONTOUR UNSTABLE"
    previous_mode = mode
    last_reliable_visibility_mode: str | None = None
    data_missing_since: float | None = None
    data_alert_logged = False
    data_alert = ""
    latest_invalid_reason = ""
    issue_tracker = IssueAccumulator()
    posture_alert = ""
    posture_data_alert = ""
    posture_unreliable_since: float | None = None
    posture_unreliable_logged = False
    reminder_store = ReminderStore()
    reminder_scheduler = ReminderScheduler(
        started_at=datetime.now().astimezone(),
        state=reminder_store.load(),
    )
    reminder_service = ReminderService(reminder_scheduler, reminder_store)
    reminder_service.publish_hydration_status(control)
    activity_service = ActivityMonitorService()
    water_button_requested = threading.Event()
    next_reminder_poll = 0.0
    next_report_poll = 0.0
    report_scheduler = AutoReportScheduler(datetime.now().astimezone())

    def blink_statistics() -> dict:
        observation_seconds = (
            completed_eye_observation_seconds
            + blink_rate_monitor.effective_time
        )
        return {
            "blink_count": blink_count,
            "valid_observation_seconds": observation_seconds,
            "average_rate_per_minute": (
                blink_count * 60.0 / observation_seconds
                if observation_seconds > 0
                else None
            ),
            "low_rate_alert_count": blink_rate_monitor.alert_count,
            "long_closure_count": long_closure_count,
            "long_closure_seconds": long_closure_seconds,
            "eye_sample_count": eye_sample_count,
        }

    def close_monitor_day(_target, rollover_at: datetime) -> None:
        nonlocal log
        nonlocal session_started
        nonlocal issue_tracker
        nonlocal blink_detector
        nonlocal blink_rate_monitor
        nonlocal blink_count
        nonlocal long_closure_count
        nonlocal long_closure_seconds
        nonlocal eye_sample_count
        nonlocal completed_eye_observation_seconds
        nonlocal activity_service

        rollover_delay = max(
            0.0,
            (datetime.now().astimezone() - rollover_at).total_seconds(),
        )
        finished_at = time.monotonic() - rollover_delay
        previous_day_end = rollover_at - timedelta(seconds=1)
        for issue_event in issue_tracker.finish(finished_at):
            event_name = issue_event.pop("event")
            log.write(event_name, occurred_at=previous_day_end, **issue_event)
        log.write(
            "session_summary",
            occurred_at=previous_day_end,
            session_seconds=finished_at - session_started,
            posture_issues=issue_tracker.summary(),
            blink_statistics=blink_statistics(),
        )
        log.write("session_ended", occurred_at=previous_day_end, reason="day_rollover")
        activity_service.finish(rollover_at)

        log = SessionLog(rollover_at)
        log.write(
            "session_started",
            occurred_at=rollover_at,
            monitor_version=APP_VERSION,
            reason="day_rollover",
        )
        session_started = finished_at
        issue_tracker = IssueAccumulator()
        blink_detector = BlinkDetector()
        blink_rate_monitor = BlinkRateMonitor()
        blink_count = 0
        long_closure_count = 0
        long_closure_seconds = 0.0
        eye_sample_count = 0
        completed_eye_observation_seconds = 0.0
        activity_service = ActivityMonitorService(
            tracker=ContinuousUseTracker(not_before=rollover_at)
        )

    try:
        for report_date in report_scheduler.generate_previous_day_if_missing(
            datetime.now().astimezone()
        ):
            log.write("daily_report_generated", report_date=report_date.isoformat())
    except Exception as error:
        log.write("daily_report_failed", message=str(error))

    print("Loading face and segmentation models; this can take up to two minutes...", flush=True)
    segmentation_worker = LatestFrameWorker(SegmentationProcessor)
    with vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:
        camera = open_camera()
        if not camera.isOpened():
            raise RuntimeError("Could not open the camera with Media Foundation.")
        try:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 640, 480)
            display_size = [640, 480]

            def handle_mouse(event, x, y, _flags, _parameter) -> None:
                if event == cv2.EVENT_LBUTTONUP and point_in_rect(
                    x,
                    y,
                    scaled_ui_rect(WATER_BUTTON_RECT, tuple(display_size)),
                ):
                    water_button_requested.set()

            cv2.setMouseCallback(WINDOW_NAME, handle_mouse)
            while True:
                if control is not None:
                    control_state = control.snapshot()
                    if control_state.exit_requested:
                        break
                    if control_state.window_visible != window_visible:
                        if set_native_window_visible(
                            WINDOW_NAME,
                            control_state.window_visible,
                        ):
                            window_visible = control_state.window_visible
                    if control_state.camera_paused != paused:
                        if control_state.camera_paused:
                            camera.release()
                            paused = True
                            data_missing_since = None
                            data_alert_logged = False
                            data_alert = ""
                        else:
                            camera = open_camera()
                            paused = not camera.isOpened()
                            control.set_camera_paused(paused)
                            rolling.clear()
                            quality_history.clear()
                            category_mask_cache = None
                            segmentation_result_sequence = -1
                        log.write("camera_paused" if paused else "camera_resumed")
                reminder_poll_now = time.monotonic()
                if reminder_poll_now >= next_reminder_poll:
                    tick_at = datetime.now().astimezone()
                    if water_button_requested.is_set():
                        water_button_requested.clear()
                        reminder_service.record_water(tick_at)
                        log.write(
                            "water_recorded_from_window",
                            amount_ml=DEFAULT_WATER_ML,
                        )
                    for amount_ml in reminder_service.consume_water_requests(
                        control,
                        tick_at,
                    ):
                        log.write(
                            "water_recorded_from_tray",
                            amount_ml=amount_ml,
                        )
                    reminder_service.tick(tick_at)
                    reminder_service.publish_hydration_status(control)
                    if reminder_poll_now >= next_report_poll:
                        try:
                            generated_reports = report_scheduler.poll(
                                tick_at,
                                close_monitor_day,
                            )
                            for report_date in generated_reports:
                                log.write(
                                    "daily_report_generated",
                                    report_date=report_date.isoformat(),
                                )
                            next_report_poll = reminder_poll_now + 1.0
                        except Exception as error:
                            log.write("daily_report_failed", message=str(error))
                            next_report_poll = reminder_poll_now + 60.0
                    activity_service.tick(tick_at)
                    next_reminder_poll = reminder_poll_now + REMINDER_POLL_SECONDS
                if paused:
                    frame = np.full((480, 640, 3), (22, 26, 30), dtype=np.uint8)
                else:
                    ok, frame = camera.read()
                    if not ok:
                        camera.release()
                        paused = True
                        if control is not None:
                            control.set_camera_paused(True)
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    video_timestamp_ms = next_video_timestamp_ms(
                        time.monotonic() - video_started,
                        video_timestamp_ms,
                    )
                    timestamp_ms = video_timestamp_ms
                    face_result = face_landmarker.detect_for_video(image, timestamp_ms)
                    face = face_result.face_landmarks[0] if face_result.face_landmarks else None
                    current_eye_metrics = (
                        eye_metrics(face, face_result) if face is not None else None
                    )
                    if current_eye_metrics is not None:
                        eye_sample_count += 1

                    segmentation_worker.submit((rgb, timestamp_ms))
                    latest_sequence, latest_mask = segmentation_worker.latest()
                    segmentation_fresh = (
                        latest_mask is not None
                        and latest_sequence != segmentation_result_sequence
                    )
                    if segmentation_fresh:
                        category_mask_cache = latest_mask
                        segmentation_result_sequence = latest_sequence
                    category_mask = category_mask_cache

                    metrics = None
                    neck_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    clearance_ratio = 0.0
                    if face is not None and category_mask is not None:
                        metrics, neck_mask = upper_body_metrics(category_mask, face)
                        if metrics:
                            clearance_ratio = metrics["head_clearance_ratio"]

                    quality_history.append((clearance_ratio, metrics is not None))
                    smooth_clearance = float(np.median([item[0] for item in quality_history]))
                    contour_fraction = float(np.mean([item[1] for item in quality_history]))
                    detected_mode = stabilized_visibility_mode(
                        previous_mode,
                        smooth_clearance,
                        contour_fraction,
                    )
                    mode = resolve_visibility_mode(
                        detected_mode,
                        last_reliable_visibility_mode,
                        metrics is not None,
                        face is not None,
                        smooth_clearance,
                    )
                    if mode != previous_mode:
                        rolling.clear()
                        log.write("visibility_changed", previous=previous_mode, current=mode)
                        previous_mode = mode
                    if mode in ("PARTIAL", "FULL", "TOO LOW"):
                        last_reliable_visibility_mode = mode

                    valid = metrics is not None and mode in ("PARTIAL", "FULL")
                    if valid:
                        invalid_reason = ""
                    elif mode == "TOO LOW" and face is not None:
                        invalid_reason = "head_too_low"
                    elif face is None:
                        invalid_reason = "face_missing"
                    else:
                        invalid_reason = "contour_unstable"

                    now = time.monotonic()
                    if valid:
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
                        latest_invalid_reason = invalid_reason
                        if data_missing_since is None:
                            data_missing_since = now
                        missing_seconds = now - data_missing_since
                        if missing_seconds >= DATA_ALERT_SECONDS:
                            data_alert = data_issue_message(invalid_reason)
                            if not data_alert_logged:
                                log.write(
                                    "data_insufficient_alert",
                                    missing_seconds=missing_seconds,
                                    reason=invalid_reason,
                                    message=data_alert,
                                )
                                data_alert_logged = True

                    if preparing_since is not None:
                        remaining = PREPARE_SECONDS - (now - preparing_since)
                        display_state = "PREPARE TO CALIBRATE"
                        status = f"Sit normally and hold still: {max(0.0, remaining):.1f}s"
                        if remaining <= 0:
                            preparing_since = None
                            calibrating_since = now
                            calibration_samples = []
                            eye_calibration_samples = []

                    if calibrating_since is not None:
                        elapsed = now - calibrating_since
                        if valid and metrics and segmentation_fresh:
                            calibration_samples.append(metrics)
                        if current_eye_metrics:
                            eye_calibration_samples.append(current_eye_metrics)
                        display_state = "CALIBRATING"
                        if elapsed < CALIBRATION_SECONDS:
                            status = (
                                "Collecting posture and open eyes: "
                                f"{max(0.0, CALIBRATION_SECONDS - elapsed):.1f}s"
                            )
                        else:
                            status = (
                                "Calibration samples - posture "
                                f"{len(calibration_samples)}/{MIN_INTEGRATED_CALIBRATION_SAMPLES}, "
                                "eyes "
                                f"{len(eye_calibration_samples)}/{MIN_INTEGRATED_CALIBRATION_SAMPLES}"
                            )
                        samples_ready = (
                            len(calibration_samples)
                            >= MIN_INTEGRATED_CALIBRATION_SAMPLES
                            and len(eye_calibration_samples)
                            >= MIN_INTEGRATED_CALIBRATION_SAMPLES
                        )
                        calibration_timed_out = elapsed >= MAX_CALIBRATION_SECONDS
                        if (
                            elapsed >= CALIBRATION_SECONDS
                            and (samples_ready or calibration_timed_out)
                        ):
                            baseline, message = build_baseline(calibration_samples)
                            blink_baseline, eye_message = build_open_eye_baseline(
                                eye_calibration_samples,
                                minimum_samples=MIN_INTEGRATED_CALIBRATION_SAMPLES,
                            )
                            calibrating_since = None
                            rolling.clear()
                            blink_detector.reset()
                            completed_eye_observation_seconds += (
                                blink_rate_monitor.effective_time
                            )
                            blink_rate_monitor = BlinkRateMonitor()
                            if baseline:
                                posture_state = "NECK NORMAL"
                                display_state = posture_state
                                status = "Monitoring relative head and upper body motion"
                                motion = "STABLE"
                                log.write(
                                    "calibrated",
                                    baseline=baseline,
                                    blink_baseline=blink_baseline,
                                    posture_sample_count=len(calibration_samples),
                                    eye_sample_count=len(eye_calibration_samples),
                                )
                                if blink_baseline is None:
                                    status = "Posture calibrated; blink calibration failed - press C again"
                                    log.write(
                                        "blink_calibration_failed",
                                        reason=eye_message,
                                        sample_count=len(eye_calibration_samples),
                                    )
                            else:
                                display_state = "CALIBRATION FAILED"
                                status = (
                                    f"Posture calibration failed: {message}; "
                                    "press C again"
                                )
                                log.write(
                                    "calibration_failed",
                                    reason=message,
                                    posture_sample_count=len(calibration_samples),
                                    eye_sample_count=len(eye_calibration_samples),
                                    blink_calibrated=blink_baseline is not None,
                                )
                                if blink_baseline is not None:
                                    log.write(
                                        "blink_calibrated",
                                        blink_baseline=blink_baseline,
                                        eye_sample_count=len(eye_calibration_samples),
                                    )

                    elif baseline and preparing_since is None:
                        if valid and metrics and segmentation_fresh:
                            rolling.append((now, metrics))
                        while rolling and now - rolling[0][0] > CURRENT_WINDOW_SECONDS:
                            rolling.popleft()

                        if rolling and now - rolling[0][0] >= CURRENT_WINDOW_SECONDS * 0.85:
                            samples = [item[1] for item in rolling]
                            current_median = median_metrics(samples)
                            raw_state, features, motion = assess_posture(
                                baseline,
                                current_median,
                                posture_state,
                                partial_view=mode == "PARTIAL",
                            )

                            if raw_state == posture_state:
                                pending_state = None
                            elif pending_state != raw_state:
                                pending_state = raw_state
                                pending_since = now
                            elif now - pending_since >= STATE_CONFIRM_SECONDS:
                                previous = posture_state
                                posture_state = raw_state
                                pending_state = None
                                log.write(
                                    "posture_changed",
                                    previous=previous,
                                    current=posture_state,
                                    motion=motion,
                                    features=features,
                                    metrics=current_median,
                                )

                            unreliable_motion = motion in (
                                "UNCERTAIN",
                                "DATA INSUFFICIENT",
                            )
                            if unreliable_motion:
                                if posture_unreliable_since is None:
                                    posture_unreliable_since = now
                                unreliable_seconds = now - posture_unreliable_since
                                if unreliable_seconds >= DATA_ALERT_SECONDS:
                                    posture_data_alert = (
                                        "POSTURE DATA UNRELIABLE - SIT UP OR RECALIBRATE"
                                    )
                                    if not posture_unreliable_logged:
                                        log.write(
                                            "posture_data_insufficient_alert",
                                            duration_seconds=unreliable_seconds,
                                            motion=motion,
                                            features=features,
                                        )
                                        posture_unreliable_logged = True
                            else:
                                if posture_unreliable_since is not None and posture_unreliable_logged:
                                    log.write(
                                        "posture_data_recovered",
                                        duration_seconds=now - posture_unreliable_since,
                                    )
                                posture_unreliable_since = None
                                posture_unreliable_logged = False
                                posture_data_alert = ""

                            if pending_state:
                                display_state = (
                                    "HEAD FORWARD CANDIDATE"
                                    if pending_state == "NECK FORWARD"
                                    else "CHECKING RECOVERY"
                                )
                                status = f"Confirming {pending_state.lower()}"
                            elif unreliable_motion:
                                display_state = "DATA INSUFFICIENT"
                                status = "Posture signals conflict; previous state is not displayed"
                            else:
                                display_state = posture_state
                                status = "Neck posture only; screen distance ignored"
                        elif not valid:
                            display_state = (
                                "HEAD TOO LOW"
                                if invalid_reason == "head_too_low"
                                else "DATA INSUFFICIENT"
                            )
                            status = data_issue_message(invalid_reason)

                    if (
                        not baseline
                        and calibrating_since is None
                        and preparing_since is None
                        and display_state != "CALIBRATION FAILED"
                    ):
                        if blink_baseline is not None:
                            display_state = "BLINK ONLY"
                            status = "Blink calibrated; press C to retry posture calibration"
                        else:
                            display_state = "NOT CALIBRATED"
                            status = "Press C and hold your normal comfortable posture"

                    blink_detected = False
                    eye_observation_valid = (
                        blink_baseline is not None
                        and calibrating_since is None
                        and preparing_since is None
                        and current_eye_metrics is not None
                    )
                    if eye_observation_valid:
                        left_open, right_open = normalized_openness(
                            current_eye_metrics,
                            blink_baseline,
                        )
                        blink_events = blink_detector.update(
                            now,
                            left_open,
                            right_open,
                            current_eye_metrics["left_blink_score"],
                            current_eye_metrics["right_blink_score"],
                        )
                        for blink_event in blink_events:
                            if blink_event.kind == "blink":
                                blink_count += 1
                                blink_detected = True
                            elif blink_event.kind == "long_eye_closure_ended":
                                long_closure_count += 1
                                long_closure_seconds += (
                                    blink_event.duration_seconds or 0.0
                                )
                            log.write(
                                blink_event.kind,
                                duration_seconds=blink_event.duration_seconds,
                            )
                    else:
                        blink_detector.update(now, None, None)

                    rate_events = blink_rate_monitor.update(
                        now,
                        valid=eye_observation_valid,
                        blink=blink_detected,
                    )
                    for rate_event in rate_events:
                        event_name = rate_event.pop("event")
                        log.write(event_name, **rate_event)
                        if event_name == "low_blink_rate_alert":
                            blink_notifier.show_low_blink_rate()

                    observed_issue = None
                    if baseline and calibrating_since is None and preparing_since is None:
                        if display_state == "HEAD TOO LOW":
                            observed_issue = "head_too_low"
                        elif (
                            valid
                            and posture_state == "NECK FORWARD"
                            and motion not in ("UNCERTAIN", "DATA INSUFFICIENT")
                        ):
                            observed_issue = "neck_forward"
                    for issue_event in issue_tracker.update(now, observed_issue):
                        event_name = issue_event.pop("event")
                        log.write(event_name, **issue_event)
                        if event_name == "posture_alert":
                            try:
                                posture_notifier.show(issue_event["issue"])
                                log.write(
                                    "posture_popup_requested",
                                    issue=issue_event["issue"],
                                )
                            except Exception as error:
                                log.write(
                                    "posture_popup_failed",
                                    issue=issue_event["issue"],
                                    message=str(error),
                                )

                    if issue_tracker.current_issue and issue_tracker.alerted:
                        posture_alert = ISSUE_MESSAGES[issue_tracker.current_issue]
                        next_alert_seconds = issue_tracker.seconds_until_next_alert(now)
                        if next_alert_seconds is not None:
                            status = (
                                "Next posture reminder in "
                                f"{format_reminder_countdown(next_alert_seconds)}"
                            )
                    else:
                        posture_alert = ""
                    if issue_tracker.current_issue and not issue_tracker.alerted:
                        issue_seconds = issue_tracker.active_duration(now)
                        status = (
                            f"Observing posture: {issue_seconds:.0f}s / "
                            f"{issue_tracker.alert_threshold():.0f}s before reminder"
                        )

                    if display_state != last_display_state:
                        log.write(
                            "state_changed",
                            previous=last_display_state,
                            current=display_state,
                            motion=motion,
                            features=features,
                            metrics=current_median,
                        )
                        last_display_state = display_state

                    tint = np.zeros_like(frame)
                    tint[:, :, 1] = neck_mask
                    cv2.addWeighted(tint, 0.16, frame, 1.0, 0, frame)
                    hydration_elapsed = (
                        hydration_elapsed_seconds(control.snapshot())
                        if control is not None
                        else max(
                            0.0,
                            (
                                datetime.now().astimezone()
                                - (
                                    reminder_scheduler.state.last_water_at
                                    or reminder_scheduler.started_at
                                )
                            ).total_seconds(),
                        )
                    )
                    display_size[:] = current_display_size(
                        WINDOW_NAME,
                        (frame.shape[1], frame.shape[0]),
                    )
                    display_frame = cv2.resize(
                        frame,
                        tuple(display_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    if face is not None:
                        draw_face(display_frame, face)
                    draw_panel(
                        display_frame,
                        display_state,
                        current_median,
                        features,
                        motion,
                        mode,
                        status,
                        posture_alert or posture_data_alert or data_alert,
                        blink_count,
                        blink_rate_monitor.rate_per_minute(),
                        hydration_elapsed=hydration_elapsed,
                        hydration_overdue=(
                            hydration_elapsed is not None
                            and hydration_elapsed
                            >= reminder_scheduler.hydration_interval.total_seconds()
                        ),
                    )
                    frame = display_frame

                if paused:
                    display_size[:] = current_display_size(
                        WINDOW_NAME,
                        (frame.shape[1], frame.shape[0]),
                    )
                    frame = cv2.resize(
                        frame,
                        tuple(display_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    draw_paused_message(frame)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c") and not paused:
                    preparing_since = time.monotonic()
                    calibrating_since = None
                    baseline = None
                    blink_baseline = None
                    eye_calibration_samples = []
                    blink_detector.reset()
                    completed_eye_observation_seconds += (
                        blink_rate_monitor.effective_time
                    )
                    blink_rate_monitor = BlinkRateMonitor()
                    rolling.clear()
                    current_median = None
                    features = {}
                    motion = "WAITING"
                    pending_state = None
                    posture_unreliable_since = None
                    posture_unreliable_logged = False
                    posture_data_alert = ""
                if key == ord("p"):
                    if paused:
                        camera = open_camera()
                        paused = not camera.isOpened()
                        rolling.clear()
                        quality_history.clear()
                        category_mask_cache = None
                        segmentation_result_sequence = -1
                    else:
                        camera.release()
                        paused = True
                        data_missing_since = None
                        data_alert_logged = False
                        data_alert = ""
                    log.write("camera_paused" if paused else "camera_resumed")
                    if control is not None:
                        control.set_camera_paused(paused)
                if key == ord("w"):
                    reminder_service.record_water(datetime.now().astimezone())
                    reminder_service.publish_hydration_status(control)
                    log.write(
                        "water_recorded_from_keyboard",
                        amount_ml=DEFAULT_WATER_ML,
                    )
                if key in (ord("y"), ord("n")):
                    log.write(
                        "user_feedback",
                        state=display_state,
                        verdict="correct" if key == ord("y") else "wrong",
                        motion=motion,
                        features=features,
                        metrics=current_median,
                    )
        finally:
            finished_at = time.monotonic()
            segmentation_worker.close()
            activity_service.finish(datetime.now().astimezone())
            for issue_event in issue_tracker.finish(finished_at):
                event_name = issue_event.pop("event")
                log.write(event_name, **issue_event)
            log.write(
                "session_summary",
                session_seconds=finished_at - session_started,
                posture_issues=issue_tracker.summary(),
                blink_statistics=blink_statistics(),
            )
            camera.release()
            cv2.destroyAllWindows()
            log.write("session_ended")


if __name__ == "__main__":
    raise SystemExit(run_with_instance_lock(main))
