import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tray_control import AppControl, build_icon_image, latest_report  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
