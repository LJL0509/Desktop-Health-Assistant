import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Callable

from activity_monitor import ActivityMonitorService, ContinuousUseTracker
from daily_report import AutoReportScheduler
from instance_lock import run_with_instance_lock


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "reminders"
STATE_PATH = DATA_DIR / "state.json"
APP_NAME = "Desktop Health Assistant"
DEFAULT_HYDRATION_MINUTES = 60
DEFAULT_WATER_ML = 250
DEFAULT_SLEEP_TIME = clock_time(1, 0)
HYDRATION_TITLE = "补水与休息提示"
HYDRATION_MESSAGE = "已经一段时间没有记录喝水了。喝点水，也离开电脑活动几分钟。"
SLEEP_TITLE = "睡眠提示"
SLEEP_MESSAGE = "哪个有意思，你是觉得现在睡觉然后明天再干事情有意思还是现在熬夜明天躺着有意思？"


def encode_datetime(value: datetime | None) -> str | None:
    return value.isoformat(timespec="seconds") if value else None


def decode_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


@dataclass
class ReminderState:
    last_water_at: datetime | None = None
    last_hydration_reminder_at: datetime | None = None
    last_sleep_reminder_date: date | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "last_water_at": encode_datetime(self.last_water_at),
            "last_hydration_reminder_at": encode_datetime(
                self.last_hydration_reminder_at
            ),
            "last_sleep_reminder_date": (
                self.last_sleep_reminder_date.isoformat()
                if self.last_sleep_reminder_date
                else None
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ReminderState":
        sleep_date = payload.get("last_sleep_reminder_date")
        return cls(
            last_water_at=decode_datetime(payload.get("last_water_at")),
            last_hydration_reminder_at=decode_datetime(
                payload.get("last_hydration_reminder_at")
            ),
            last_sleep_reminder_date=(
                date.fromisoformat(sleep_date) if sleep_date else None
            ),
        )


@dataclass(frozen=True)
class ReminderEvent:
    kind: str
    occurred_at: datetime
    title: str
    message: str


class ReminderScheduler:
    def __init__(
        self,
        started_at: datetime,
        state: ReminderState | None = None,
        hydration_interval: timedelta = timedelta(
            minutes=DEFAULT_HYDRATION_MINUTES
        ),
        sleep_time: clock_time = DEFAULT_SLEEP_TIME,
    ) -> None:
        if started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        if hydration_interval.total_seconds() <= 0:
            raise ValueError("hydration_interval must be positive")
        self.state = state or ReminderState()
        self.hydration_interval = hydration_interval
        self.sleep_time = sleep_time
        self.last_checked_at = started_at
        self.started_at = started_at

    def _hydration_anchor(self) -> datetime:
        candidates = [
            value
            for value in (
                self.state.last_water_at,
                self.state.last_hydration_reminder_at,
            )
            if value is not None
        ]
        return max(candidates) if candidates else self.started_at

    def next_hydration_at(self) -> datetime:
        return self._hydration_anchor() + self.hydration_interval

    def record_water(self, now: datetime) -> None:
        self._validate_now(now)
        self.state.last_water_at = now

    def poll(self, now: datetime) -> list[ReminderEvent]:
        self._validate_now(now)
        if now < self.last_checked_at:
            self.last_checked_at = now
            return []

        events: list[ReminderEvent] = []
        if now >= self.next_hydration_at():
            self.state.last_hydration_reminder_at = now
            events.append(
                ReminderEvent(
                    kind="hydration_reminder",
                    occurred_at=now,
                    title=HYDRATION_TITLE,
                    message=HYDRATION_MESSAGE,
                )
            )

        sleep_target = datetime.combine(
            now.date(),
            self.sleep_time,
            tzinfo=now.tzinfo,
        )
        crossed_sleep_time = self.last_checked_at < sleep_target <= now
        already_reminded = self.state.last_sleep_reminder_date == now.date()
        if crossed_sleep_time and not already_reminded:
            self.state.last_sleep_reminder_date = now.date()
            events.append(
                ReminderEvent(
                    kind="sleep_reminder",
                    occurred_at=now,
                    title=SLEEP_TITLE,
                    message=SLEEP_MESSAGE,
                )
            )

        self.last_checked_at = now
        return events

    @staticmethod
    def _validate_now(now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")


class ReminderStore:
    def __init__(self, state_path: Path = STATE_PATH) -> None:
        self.state_path = state_path
        self.lock = threading.Lock()

    def load(self) -> ReminderState:
        with self.lock:
            if not self.state_path.exists():
                return ReminderState()
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return ReminderState.from_dict(payload)

    def save(self, state: ReminderState) -> None:
        with self.lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.state_path)

    def log(self, event: str, occurred_at: datetime, **details) -> None:
        with self.lock:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / f"{occurred_at.date().isoformat()}.jsonl"
            payload = {
                "timestamp": occurred_at.isoformat(timespec="seconds"),
                "event": event,
                **details,
            }
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")


class WindowsNotifier:
    def __init__(
        self,
        on_water_recorded: Callable[[datetime, int], None],
        on_action: Callable[[str, datetime], None],
    ) -> None:
        from windows_toasts import InteractableWindowsToaster

        self.toaster = InteractableWindowsToaster(APP_NAME)
        self.on_water_recorded = on_water_recorded
        self.on_action = on_action

    def show(self, event: ReminderEvent) -> None:
        from windows_toasts import (
            Toast,
            ToastAudio,
            ToastButton,
            ToastDuration,
            ToastScenario,
        )

        actions = []
        if event.kind == "hydration_reminder":
            actions.append(ToastButton("已喝水", "record_water"))
        elif event.kind == "sleep_reminder":
            actions.append(ToastButton("知道了", "sleep_acknowledged"))

        def activated(args) -> None:
            now = datetime.now().astimezone()
            if args.arguments == "record_water":
                self.on_water_recorded(now, DEFAULT_WATER_ML)
            else:
                self.on_action(args.arguments or "notification_opened", now)

        toast = Toast(
            text_fields=[event.title, event.message],
            audio=ToastAudio(silent=True),
            duration=ToastDuration.Long,
            scenario=ToastScenario.Reminder,
            actions=actions,
            on_activated=activated,
        )
        self.toaster.show_toast(toast)


class ReminderService:
    def __init__(
        self,
        scheduler: ReminderScheduler,
        store: ReminderStore,
    ) -> None:
        self.scheduler = scheduler
        self.store = store
        self.lock = threading.Lock()
        self.notifier = WindowsNotifier(
            on_water_recorded=self.record_water,
            on_action=self.record_action,
        )

    def record_water(self, now: datetime, amount_ml: int = DEFAULT_WATER_ML) -> None:
        with self.lock:
            self.scheduler.record_water(now)
            self.store.save(self.scheduler.state)
            self.store.log("water_recorded", now, amount_ml=amount_ml)

    def record_action(self, action: str, now: datetime) -> None:
        self.store.log("reminder_action", now, action=action)

    def tick(self, now: datetime) -> list[ReminderEvent]:
        with self.lock:
            events = self.scheduler.poll(now)
            if events:
                self.store.save(self.scheduler.state)
            for event in events:
                self.store.log(
                    event.kind,
                    event.occurred_at,
                    title=event.title,
                    message=event.message,
                )
                self.notifier.show(event)
            return events


def parse_sleep_time(value: str) -> clock_time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise argparse.ArgumentTypeError("sleep time must use HH:MM") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desktop Health Assistant reminders")
    parser.add_argument(
        "--hydration-minutes",
        type=float,
        default=DEFAULT_HYDRATION_MINUTES,
    )
    parser.add_argument(
        "--sleep-time",
        type=parse_sleep_time,
        default=DEFAULT_SLEEP_TIME,
    )
    parser.add_argument(
        "--record-water",
        type=int,
        metavar="ML",
        help="record water and exit",
    )
    parser.add_argument(
        "--test-notifications",
        action="store_true",
        help="show hydration and sleep notifications immediately",
    )
    parser.add_argument(
        "--test-kind",
        choices=("all", "hydration", "sleep"),
        default="all",
        help="select which test notification to show",
    )
    return parser


def main(argv: list[str] | None = None, control=None) -> None:
    args = build_parser().parse_args(argv)
    now = datetime.now().astimezone()
    store = ReminderStore()
    scheduler = ReminderScheduler(
        started_at=now,
        state=store.load(),
        hydration_interval=timedelta(minutes=args.hydration_minutes),
        sleep_time=args.sleep_time,
    )
    service = ReminderService(scheduler, store)

    if args.record_water is not None:
        service.record_water(now, args.record_water)
        print(f"Recorded {args.record_water} ml at {now.isoformat(timespec='seconds')}")
        return

    if args.test_notifications:
        all_test_events = (
            ReminderEvent(
                "hydration_reminder",
                now,
                HYDRATION_TITLE,
                HYDRATION_MESSAGE,
            ),
            ReminderEvent(
                "sleep_reminder",
                now,
                SLEEP_TITLE,
                SLEEP_MESSAGE,
            ),
        )

        def record_test_water(action_at: datetime, amount_ml: int) -> None:
            store.log(
                "test_notification_action",
                action_at,
                action="record_water",
                amount_ml=amount_ml,
            )

        def record_test_action(action: str, action_at: datetime) -> None:
            store.log(
                "test_notification_action",
                action_at,
                action=action,
            )

        test_notifier = WindowsNotifier(record_test_water, record_test_action)
        test_events = [
            event
            for event in all_test_events
            if args.test_kind == "all"
            or event.kind.startswith(args.test_kind)
        ]
        for index, event in enumerate(test_events):
            if index:
                time.sleep(4)
            test_notifier.show(event)
            store.log(
                "test_notification",
                now,
                kind=event.kind,
                title=event.title,
                message=event.message,
            )
        print("Test notifications sent; waiting 30 seconds for button actions...")
        time.sleep(30)
        return

    print(
        "Reminder service running. "
        f"Hydration: {args.hydration_minutes:g} min; "
        f"sleep: {args.sleep_time.strftime('%H:%M')}; "
        "continuous-use statistics enabled. Press Ctrl+C to stop."
    )
    activity_service = ActivityMonitorService()
    report_scheduler = AutoReportScheduler(tick_at := datetime.now().astimezone())
    try:
        generated = report_scheduler.generate_previous_day_if_missing(tick_at)
        for report_date in generated:
            print(f"Generated missing daily report: {report_date.isoformat()}")
    except Exception as error:
        print(f"Daily report generation failed; will retry: {error}")

    def close_activity_day(_target: date, rollover_at: datetime) -> None:
        nonlocal activity_service
        activity_service.finish(rollover_at)
        activity_service = ActivityMonitorService(
            tracker=ContinuousUseTracker(not_before=rollover_at)
        )

    next_report_poll = 0.0
    try:
        while True:
            if control is not None and control.snapshot().exit_requested:
                print("Exit requested from system tray.")
                break
            tick_at = datetime.now().astimezone()
            service.tick(tick_at)
            loop_now = time.monotonic()
            if loop_now >= next_report_poll:
                try:
                    for report_date in report_scheduler.poll(
                        tick_at,
                        close_activity_day,
                    ):
                        print(f"Generated daily report: {report_date.isoformat()}")
                    next_report_poll = loop_now + 1.0
                except Exception as error:
                    print(f"Daily report generation failed; will retry: {error}")
                    next_report_poll = loop_now + 60.0
            activity_service.tick(tick_at)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Reminder service stopped.")
    finally:
        activity_service.finish(datetime.now().astimezone())


if __name__ == "__main__":
    raise SystemExit(run_with_instance_lock(main))
