import json
import sys
import tempfile
import threading
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reminder_service import (  # noqa: E402
    HYDRATION_MESSAGE,
    ReminderScheduler,
    ReminderService,
    ReminderState,
    ReminderStore,
    SLEEP_MESSAGE,
    WindowsNotifier,
)


CHINA = timezone(timedelta(hours=8))


def at(hour: int, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=CHINA)


class ReminderSchedulerTest(unittest.TestCase):
    def test_fresh_start_waits_full_hydration_interval(self) -> None:
        scheduler = ReminderScheduler(at(9), hydration_interval=timedelta(minutes=60))
        self.assertEqual(scheduler.poll(at(9, 59)), [])
        events = scheduler.poll(at(10))
        self.assertEqual([event.kind for event in events], ["hydration_reminder"])
        self.assertIn("离开电脑活动", events[0].message)
        self.assertEqual(events[0].message, HYDRATION_MESSAGE)

    def test_hydration_reminder_repeats_only_after_another_interval(self) -> None:
        scheduler = ReminderScheduler(at(9), hydration_interval=timedelta(minutes=60))
        scheduler.poll(at(10))
        self.assertEqual(scheduler.poll(at(10, 59)), [])
        self.assertEqual(
            [event.kind for event in scheduler.poll(at(11))],
            ["hydration_reminder"],
        )

    def test_recording_water_resets_hydration_timer(self) -> None:
        scheduler = ReminderScheduler(at(9), hydration_interval=timedelta(minutes=60))
        scheduler.record_water(at(9, 45))
        self.assertEqual(scheduler.poll(at(10)), [])
        self.assertEqual(
            [event.kind for event in scheduler.poll(at(10, 45))],
            ["hydration_reminder"],
        )

    def test_sleep_reminder_fires_when_running_across_one_am(self) -> None:
        scheduler = ReminderScheduler(at(0, 59), sleep_time=time(1, 0))
        events = scheduler.poll(at(1, 0))
        self.assertEqual([event.kind for event in events], ["sleep_reminder"])
        self.assertEqual(events[0].message, SLEEP_MESSAGE)

    def test_sleep_reminder_is_not_backfilled_after_late_start(self) -> None:
        scheduler = ReminderScheduler(at(1, 30), sleep_time=time(1, 0))
        self.assertEqual(scheduler.poll(at(2, 0)), [])

    def test_sleep_reminder_fires_only_once_each_day(self) -> None:
        state = ReminderState(last_sleep_reminder_date=date(2026, 8, 11))
        scheduler = ReminderScheduler(at(0, 59), state=state, sleep_time=time(1, 0))
        self.assertEqual(scheduler.poll(at(1, 0)), [])

    def test_sleep_reminder_fires_again_next_day(self) -> None:
        state = ReminderState(last_sleep_reminder_date=date(2026, 8, 11))
        scheduler = ReminderScheduler(at(23, 59), state=state, sleep_time=time(1, 0))
        events = scheduler.poll(at(1, 0, day=12))
        sleep_events = [event for event in events if event.kind == "sleep_reminder"]
        self.assertEqual(len(sleep_events), 1)

    def test_clock_rollback_does_not_trigger_reminder(self) -> None:
        scheduler = ReminderScheduler(at(9))
        self.assertEqual(scheduler.poll(at(8)), [])


class ReminderStoreTest(unittest.TestCase):
    def test_state_round_trip(self) -> None:
        state = ReminderState(
            last_water_at=at(9, 30),
            last_hydration_reminder_at=at(10, 30),
            last_sleep_reminder_date=date(2026, 8, 11),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ReminderStore(Path(directory) / "state.json")
            store.save(state)
            loaded = store.load()
        self.assertEqual(loaded, state)

    def test_saved_state_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            store = ReminderStore(path)
            store.save(ReminderState(last_water_at=at(9, 30)))
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["last_water_at"], "2026-08-11T09:30:00+08:00")


class ReminderServiceControlTest(unittest.TestCase):
    def test_consumes_tray_water_request_through_official_service(self) -> None:
        scheduler = ReminderScheduler(at(9))
        store = Mock()
        service = ReminderService.__new__(ReminderService)
        service.scheduler = scheduler
        service.store = store
        service.lock = threading.Lock()
        service.notifier = Mock()
        control = Mock()
        control.consume_water_requests.return_value = [250]

        amounts = service.consume_water_requests(control, at(9, 20))

        self.assertEqual(amounts, [250])
        self.assertEqual(scheduler.state.last_water_at, at(9, 20))
        store.save.assert_called_once_with(scheduler.state)
        store.log.assert_called_once_with("water_recorded", at(9, 20), amount_ml=250)
        control.update_hydration_status.assert_called_once()


class WindowsNotifierTest(unittest.TestCase):
    def test_uses_interactable_toaster_for_notification_buttons(self) -> None:
        notifier = WindowsNotifier(lambda *_: None, lambda *_: None)
        self.assertEqual(
            type(notifier.toaster).__name__,
            "InteractableWindowsToaster",
        )



if __name__ == "__main__":
    unittest.main()
