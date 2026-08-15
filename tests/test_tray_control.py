import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tray_control import (  # noqa: E402
    AppControl,
    build_icon_image,
    format_elapsed_time,
    hydration_elapsed_seconds,
    latest_report,
)


CHINA = timezone(timedelta(hours=8))


class AppControlTest(unittest.TestCase):
    def test_camera_commands_are_thread_safe_state_changes(self) -> None:
        control = AppControl(camera_enabled=True)
        self.assertTrue(control.snapshot().window_visible)
        self.assertFalse(control.snapshot().camera_paused)

        control.toggle_window_visible()
        control.toggle_camera_paused()
        snapshot = control.snapshot()
        self.assertFalse(snapshot.window_visible)
        self.assertTrue(snapshot.camera_paused)

        control.request_exit()
        self.assertTrue(control.snapshot().exit_requested)

    def test_no_camera_mode_ignores_camera_commands(self) -> None:
        control = AppControl(camera_enabled=False)
        control.toggle_window_visible()
        control.toggle_camera_paused()
        snapshot = control.snapshot()
        self.assertTrue(snapshot.window_visible)
        self.assertFalse(snapshot.camera_paused)

    def test_water_requests_are_consumed_once(self) -> None:
        control = AppControl(camera_enabled=True)
        control.request_water_record(250)
        control.request_water_record(300)
        self.assertEqual(control.consume_water_requests(), [250, 300])
        self.assertEqual(control.consume_water_requests(), [])

    def test_hydration_snapshot_uses_last_water_as_anchor(self) -> None:
        control = AppControl(camera_enabled=True)
        started = datetime(2026, 8, 12, 9, 0, tzinfo=CHINA)
        last_water = datetime(2026, 8, 12, 9, 30, tzinfo=CHINA)
        control.update_hydration_status(started, last_water, 3600)

        elapsed = hydration_elapsed_seconds(
            control.snapshot(),
            datetime(2026, 8, 12, 10, 5, tzinfo=CHINA),
        )

        self.assertEqual(elapsed, 35 * 60)

    def test_formats_hydration_elapsed_for_compact_menu(self) -> None:
        self.assertEqual(format_elapsed_time(None), "正在初始化")
        self.assertEqual(format_elapsed_time(59 * 60), "59分钟")
        self.assertEqual(format_elapsed_time(65 * 60), "1小时05分钟")


class TrayHelpersTest(unittest.TestCase):
    def test_latest_report_uses_date_named_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "2026-08-10.md").write_text("old", encoding="utf-8")
            expected = root / "2026-08-12.md"
            expected.write_text("new", encoding="utf-8")
            (root / "notes.md").write_text("ignore", encoding="utf-8")

            report = latest_report(root)

        self.assertEqual(report, expected)

    def test_latest_report_returns_none_when_directory_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(latest_report(Path(directory)))

    def test_generated_icon_has_stable_rgba_dimensions(self) -> None:
        icon = build_icon_image(48)
        self.assertEqual(icon.mode, "RGBA")
        self.assertEqual(icon.size, (48, 48))
        self.assertIsNotNone(icon.getbbox())


if __name__ == "__main__":
    unittest.main()
