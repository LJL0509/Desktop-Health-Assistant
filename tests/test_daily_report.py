import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from daily_report import (  # noqa: E402
    AutoReportScheduler,
    build_report,
    format_duration,
    render_markdown,
    write_report,
)


CHINA = timezone(timedelta(hours=8))
TARGET = date(2026, 8, 12)


def timestamp(hour: int, minute: int, second: int = 0) -> str:
    return datetime(2026, 8, 12, hour, minute, second, tzinfo=CHINA).isoformat()


def write_events(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events),
        encoding="utf-8",
    )


class DailyReportTest(unittest.TestCase):
    def make_report(self, root: Path) -> dict:
        monitor_dir = root / "monitor"
        reminder_dir = root / "reminders"
        activity_dir = root / "activity"
        write_events(
            monitor_dir / "session.jsonl",
            [
                {"timestamp": timestamp(9, 0), "event": "session_started"},
                {
                    "timestamp": timestamp(9, 2),
                    "event": "posture_issue_started",
                    "issue": "neck_forward",
                },
                {
                    "timestamp": timestamp(9, 3),
                    "event": "posture_alert",
                    "issue": "neck_forward",
                },
                {
                    "timestamp": timestamp(9, 4),
                    "event": "posture_issue_ended",
                    "issue": "neck_forward",
                    "duration_seconds": 120,
                    "alerted": True,
                },
                {
                    "timestamp": timestamp(9, 6),
                    "event": "data_recovered",
                    "missing_seconds": 30,
                },
                {"timestamp": timestamp(9, 10), "event": "camera_paused"},
                {"timestamp": timestamp(9, 12), "event": "camera_resumed"},
                {
                    "timestamp": timestamp(9, 20),
                    "event": "session_summary",
                    "session_seconds": 1200,
                    "blink_statistics": {
                        "blink_count": 80,
                        "valid_observation_seconds": 600,
                        "average_rate_per_minute": 8,
                        "low_rate_alert_count": 1,
                        "long_closure_count": 2,
                        "long_closure_seconds": 4,
                    },
                },
                {"timestamp": timestamp(9, 20), "event": "session_ended"},
            ],
        )
        write_events(
            reminder_dir / "2026-08-12.jsonl",
            [
                {"timestamp": timestamp(10, 0), "event": "hydration_reminder"},
                {"timestamp": timestamp(11, 0), "event": "hydration_reminder"},
                {
                    "timestamp": timestamp(11, 5),
                    "event": "water_recorded",
                    "amount_ml": 250,
                },
                {"timestamp": timestamp(12, 5), "event": "hydration_reminder"},
                {"timestamp": timestamp(1, 0), "event": "sleep_reminder"},
                {
                    "timestamp": timestamp(1, 1),
                    "event": "reminder_action",
                    "action": "sleep_acknowledged",
                },
                {"timestamp": timestamp(13, 0), "event": "test_notification"},
            ],
        )
        write_events(
            activity_dir / "2026-08-12.jsonl",
            [
                {"timestamp": timestamp(8, 0), "event": "continuous_use_started"},
                {
                    "timestamp": timestamp(9, 30),
                    "event": "continuous_use_ended",
                    "started_at": timestamp(8, 0),
                    "duration_seconds": 5400,
                },
            ],
        )
        return build_report(
            TARGET,
            monitor_dir=monitor_dir,
            reminder_dir=reminder_dir,
            activity_dir=activity_dir,
            generated_at=datetime(2026, 8, 12, 23, 0, tzinfo=CHINA),
        )

    def test_aggregates_monitoring_posture_and_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self.make_report(Path(directory))
        self.assertEqual(report["monitoring"]["run_seconds"], 1200)
        self.assertEqual(report["monitoring"]["camera_paused_seconds"], 120)
        self.assertEqual(report["monitoring"]["valid_monitoring_seconds"], 1050)
        self.assertAlmostEqual(report["monitoring"]["data_coverage_ratio"], 1050 / 1080)
        self.assertEqual(report["posture_issues"]["neck_forward"]["total_seconds"], 120)
        self.assertEqual(report["posture_issues"]["neck_forward"]["alert_count"], 1)
        self.assertEqual(report["reminders"]["water_record_count"], 1)
        self.assertEqual(report["reminders"]["water_total_ml"], 250)
        self.assertEqual(report["reminders"]["hydration_reminder_count"], 3)
        self.assertEqual(report["reminders"]["longest_unconfirmed_water_seconds"], 7200)
        self.assertTrue(report["reminders"]["hydration_confirmation_pending"])
        self.assertEqual(report["reminders"]["sleep_acknowledged_count"], 1)
        self.assertEqual(report["computer_activity"]["total_seconds"], 5400)
        self.assertEqual(report["computer_activity"]["longest_seconds"], 5400)
        self.assertEqual(report["blink_statistics"]["blink_count"], 80)
        self.assertEqual(report["blink_statistics"]["average_rate_per_minute"], 8)
        self.assertEqual(report["blink_statistics"]["long_closure_count"], 2)

    def test_marks_unfinished_session_as_estimated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_events(
                root / "monitor" / "session.jsonl",
                [
                    {"timestamp": timestamp(9, 0), "event": "session_started"},
                    {"timestamp": timestamp(9, 5), "event": "calibrated"},
                ],
            )
            report = build_report(
                TARGET,
                root / "monitor",
                root / "reminders",
                root / "activity",
            )
        self.assertEqual(report["monitoring"]["run_seconds"], 300)
        self.assertEqual(report["monitoring"]["complete_session_count"], 0)
        self.assertTrue(any("没有正常结束" in note for note in report["data_notes"]))

    def test_writes_readable_and_structured_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.make_report(root)
            json_path, markdown_path = write_report(report, root / "reports")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")
        self.assertEqual(payload["date"], "2026-08-12")
        self.assertIn("每日健康提示报告", markdown)
        self.assertIn("尚未接入", markdown)
        self.assertIn("不是医学诊断", markdown)

    def test_formats_duration_without_fractional_noise(self) -> None:
        self.assertEqual(format_duration(0), "0 秒")
        self.assertEqual(format_duration(65.4), "1 分钟 5 秒")
        self.assertEqual(format_duration(3660), "1 小时 1 分钟")

    def test_counts_open_activity_session_until_report_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_events(
                root / "activity" / "2026-08-12.jsonl",
                [
                    {
                        "timestamp": timestamp(20, 0),
                        "event": "continuous_use_started",
                    }
                ],
            )
            report = build_report(
                TARGET,
                root / "monitor",
                root / "reminders",
                root / "activity",
                generated_at=datetime(2026, 8, 12, 20, 30, tzinfo=CHINA),
            )
        self.assertEqual(report["computer_activity"]["total_seconds"], 1800)
        self.assertEqual(report["computer_activity"]["open_session_count"], 1)

    def test_no_data_report_does_not_claim_health_is_normal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build_report(
                TARGET,
                root / "monitor",
                root / "reminders",
                root / "activity",
                generated_at=datetime(2026, 8, 12, 23, tzinfo=CHINA),
            )
        self.assertIn("没有有效监测数据", report["guidance"][0])


class AutoReportSchedulerTest(unittest.TestCase):
    def make_scheduler(self, root: Path, started_at: datetime, calls: list) -> AutoReportScheduler:
        def generate(target: date, generated_at: datetime) -> dict:
            calls.append((target, generated_at))
            return {
                "date": target.isoformat(),
                "generated_at": generated_at.isoformat(),
                "monitoring": {
                    "session_count": 0,
                    "complete_session_count": 0,
                    "run_seconds": 0,
                    "camera_paused_seconds": 0,
                    "camera_data_missing_seconds": 0,
                    "valid_monitoring_seconds": 0,
                    "data_coverage_ratio": None,
                },
                "computer_activity": {
                    "session_count": 0,
                    "total_seconds": 0,
                    "longest_seconds": 0,
                },
                "blink_statistics": {
                    "valid_observation_seconds": 0,
                    "blink_count": 0,
                    "average_rate_per_minute": None,
                    "low_rate_alert_count": 0,
                    "long_closure_count": 0,
                    "long_closure_seconds": 0,
                },
                "posture_issues": {
                    "neck_forward": {
                        "total_seconds": 0,
                        "episode_count": 0,
                        "longest_seconds": 0,
                        "alert_count": 0,
                    },
                    "head_too_low": {
                        "total_seconds": 0,
                        "episode_count": 0,
                        "longest_seconds": 0,
                        "alert_count": 0,
                    },
                },
                "reminders": {
                    "water_record_count": 0,
                    "water_total_ml": 0,
                    "hydration_reminder_count": 0,
                    "longest_unconfirmed_water_seconds": 0,
                    "hydration_confirmation_pending": False,
                    "sleep_reminder_count": 0,
                    "sleep_acknowledged_count": 0,
                },
                "guidance": [],
                "not_yet_monitored": [],
                "data_notes": [],
                "medical_notice": "test",
            }

        return AutoReportScheduler(started_at, root / "reports", generate)

    def test_startup_generates_missing_previous_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            scheduler = self.make_scheduler(
                Path(directory),
                datetime(2026, 8, 12, 9, tzinfo=CHINA),
                calls,
            )
            generated = scheduler.generate_previous_day_if_missing(
                datetime(2026, 8, 12, 9, tzinfo=CHINA)
            )
        self.assertEqual(generated, [date(2026, 8, 11)])
        self.assertEqual([item[0] for item in calls], [date(2026, 8, 11)])

    def test_existing_report_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            scheduler = self.make_scheduler(
                root,
                datetime(2026, 8, 12, 9, tzinfo=CHINA),
                calls,
            )
            reports = root / "reports"
            reports.mkdir()
            (reports / "2026-08-11.json").write_text("old", encoding="utf-8")
            (reports / "2026-08-11.md").write_text("old", encoding="utf-8")
            generated = scheduler.generate_previous_day_if_missing(
                datetime(2026, 8, 12, 9, tzinfo=CHINA)
            )
            content = (reports / "2026-08-11.md").read_text(encoding="utf-8")
        self.assertEqual(generated, [])
        self.assertEqual(calls, [])
        self.assertEqual(content, "old")

    def test_midnight_closes_day_before_generating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            order = []
            scheduler = self.make_scheduler(
                Path(directory),
                datetime(2026, 8, 12, 23, 59, tzinfo=CHINA),
                calls,
            )

            rollover_times = []

            def close_day(target: date, now: datetime) -> None:
                order.append(("close", target))
                rollover_times.append(now)

            original_generator = scheduler.generator

            def generate(target: date, now: datetime) -> dict:
                order.append(("generate", target))
                return original_generator(target, now)

            scheduler.generator = generate
            generated = scheduler.poll(
                datetime(2026, 8, 13, 0, 0, 1, tzinfo=CHINA),
                close_day,
            )
        self.assertEqual(generated, [date(2026, 8, 12)])
        self.assertEqual(
            order,
            [("close", date(2026, 8, 12)), ("generate", date(2026, 8, 12))],
        )
        self.assertEqual(
            rollover_times,
            [datetime(2026, 8, 13, 0, 0, tzinfo=CHINA)],
        )

    def test_poll_same_day_is_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            scheduler = self.make_scheduler(
                Path(directory),
                datetime(2026, 8, 12, 9, tzinfo=CHINA),
                calls,
            )
            generated = scheduler.poll(
                datetime(2026, 8, 12, 23, tzinfo=CHINA)
            )
        self.assertEqual(generated, [])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
