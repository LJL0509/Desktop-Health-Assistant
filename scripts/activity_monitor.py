import ctypes
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from app_paths import data_path


DATA_DIR = data_path("activity")
DEFAULT_BREAK_RESET_MINUTES = 5


@dataclass(frozen=True)
class ActivityEvent:
    kind: str
    occurred_at: datetime
    details: dict = field(default_factory=dict)


class ContinuousUseTracker:
    def __init__(
        self,
        break_reset_after: timedelta = timedelta(
            minutes=DEFAULT_BREAK_RESET_MINUTES
        ),
        not_before: datetime | None = None,
    ) -> None:
        if break_reset_after.total_seconds() <= 0:
            raise ValueError("break_reset_after must be positive")
        self.break_reset_after = break_reset_after
        self.not_before = not_before
        self.started_at: datetime | None = None

    def update(self, now: datetime, idle_seconds: float) -> list[ActivityEvent]:
        self._validate(now, idle_seconds)
        last_input_at = now - timedelta(seconds=idle_seconds)
        if self.not_before is not None:
            last_input_at = max(last_input_at, self.not_before)
        events = []

        if idle_seconds >= self.break_reset_after.total_seconds():
            if self.started_at is not None:
                ended_at = max(self.started_at, min(now, last_input_at))
                events.append(self._end_event(ended_at, "input_break", idle_seconds))
            return events

        if self.started_at is None:
            self.started_at = last_input_at
            events.append(
                ActivityEvent(
                    "continuous_use_started",
                    self.started_at,
                )
            )

        return events

    def finish(self, now: datetime, idle_seconds: float = 0.0) -> list[ActivityEvent]:
        self._validate(now, idle_seconds)
        if self.started_at is None:
            return []
        last_input_at = now - timedelta(seconds=idle_seconds)
        ended_at = max(self.started_at, min(now, last_input_at))
        return [self._end_event(ended_at, "service_stopped", idle_seconds)]

    def _end_event(
        self,
        ended_at: datetime,
        reason: str,
        idle_seconds: float,
    ) -> ActivityEvent:
        if self.started_at is None:
            raise RuntimeError("Cannot end a continuous-use session before it starts")
        duration = max(0.0, (ended_at - self.started_at).total_seconds())
        event = ActivityEvent(
            "continuous_use_ended",
            ended_at,
            {
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "duration_seconds": duration,
                "reason": reason,
                "detected_idle_seconds": idle_seconds,
            },
        )
        self.started_at = None
        return event

    @staticmethod
    def _validate(now: datetime, idle_seconds: float) -> None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        if idle_seconds < 0:
            raise ValueError("idle_seconds cannot be negative")


class _LastInputInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint),
    ]


class WindowsIdleProvider:
    def __init__(self) -> None:
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("Windows idle detection is only available on Windows")
        self.user32 = ctypes.windll.user32
        self.kernel32 = ctypes.windll.kernel32

    def idle_seconds(self) -> float:
        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(info)
        if not self.user32.GetLastInputInfo(ctypes.byref(info)):
            raise ctypes.WinError()
        current_tick = int(self.kernel32.GetTickCount())
        elapsed_ms = (current_tick - int(info.dwTime)) & 0xFFFFFFFF
        return elapsed_ms / 1000.0


class ActivityStore:
    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.lock = threading.Lock()

    def log(self, event: ActivityEvent) -> None:
        with self.lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            path = self.data_dir / f"{event.occurred_at.date().isoformat()}.jsonl"
            payload = {
                "timestamp": event.occurred_at.isoformat(timespec="seconds"),
                "event": event.kind,
                **event.details,
            }
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ActivityMonitorService:
    def __init__(
        self,
        tracker: ContinuousUseTracker | None = None,
        provider: WindowsIdleProvider | None = None,
        store: ActivityStore | None = None,
    ) -> None:
        self.tracker = tracker or ContinuousUseTracker()
        self.provider = provider or WindowsIdleProvider()
        self.store = store or ActivityStore()

    def tick(self, now: datetime) -> list[ActivityEvent]:
        events = self.tracker.update(now, self.provider.idle_seconds())
        self._handle(events)
        return events

    def finish(self, now: datetime) -> list[ActivityEvent]:
        events = self.tracker.finish(now, self.provider.idle_seconds())
        self._handle(events)
        return events

    def _handle(self, events: list[ActivityEvent]) -> None:
        for event in events:
            self.store.log(event)
