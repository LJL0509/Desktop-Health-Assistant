import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from activity_monitor import ActivityStore, ContinuousUseTracker  # noqa: E402


CHINA = timezone(timedelta(hours=8))


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 8, 12, hour, minute, second, tzinfo=CHINA)


class ContinuousUseTrackerTest(unittest.TestCase):
    def make_tracker(self) -> ContinuousUseTracker:
        return ContinuousUseTracker(
            break_reset_after=timedelta(minutes=5),
        )

    def test_starts_session_when_input_is_recent(self) -> None:
        tracker = self.make_tracker()
        events = tracker.update(at(9), idle_seconds=10)
        self.assertEqual([event.kind for event in events], ["continuous_use_started"])
        self.assertEqual(events[0].occurred_at, at(8, 59, 50))

    def test_does_not_emit_a_separate_hourly_reminder(self) -> None:
        tracker = self.make_tracker()
        tracker.update(at(9), idle_seconds=0)
        self.assertEqual(tracker.update(at(10), idle_seconds=0), [])
        self.assertEqual(tracker.update(at(10, 30), idle_seconds=0), [])

    def test_five_minute_input_break_ends_and_resets_session(self) -> None:
        tracker = self.make_tracker()
        tracker.update(at(9), idle_seconds=0)
        ended = tracker.update(at(10, 5), idle_seconds=300)
        restarted = tracker.update(at(10, 6), idle_seconds=0)
        self.assertEqual([event.kind for event in ended], ["continuous_use_ended"])
        self.assertEqual(ended[0].details["duration_seconds"], 3600)
        self.assertEqual([event.kind for event in restarted], ["continuous_use_started"])

    def test_short_idle_period_does_not_reset_session(self) -> None:
        tracker = self.make_tracker()
        tracker.update(at(9), idle_seconds=0)
        events = tracker.update(at(9, 4), idle_seconds=240)
        self.assertEqual(events, [])
        self.assertEqual(tracker.started_at, at(9))

    def test_finish_closes_current_session(self) -> None:
        tracker = self.make_tracker()
        tracker.update(at(9), idle_seconds=0)
        events = tracker.finish(at(9, 20), idle_seconds=30)
        self.assertEqual(events[0].details["duration_seconds"], 1170)
        self.assertEqual(events[0].details["reason"], "service_stopped")

    def test_new_day_session_does_not_start_before_midnight_floor(self) -> None:
        floor = at(0, 0)
        tracker = ContinuousUseTracker(not_before=floor)
        events = tracker.update(at(0, 1), idle_seconds=120)
        self.assertEqual(events[0].occurred_at, floor)


class ActivityStoreTest(unittest.TestCase):
    def test_writes_only_timestamps_and_summary_events(self) -> None:
        tracker = ContinuousUseTracker()
        event = tracker.update(at(9), idle_seconds=0)[0]
        with tempfile.TemporaryDirectory() as directory:
            store = ActivityStore(Path(directory))
            store.log(event)
            payload = json.loads(
                (Path(directory) / "2026-08-12.jsonl").read_text(encoding="utf-8")
            )
        self.assertEqual(payload["event"], "continuous_use_started")
        self.assertNotIn("key", payload)


if __name__ == "__main__":
    unittest.main()
