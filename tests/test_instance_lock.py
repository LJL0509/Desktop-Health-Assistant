import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from desktop_health_assistant import build_parser, main  # noqa: E402
from instance_lock import AlreadyRunningError, SingleInstanceLock  # noqa: E402


class SingleInstanceLockTest(unittest.TestCase):
    def test_second_lock_with_same_name_is_rejected(self) -> None:
        name = f"Local\\DesktopHealthAssistant.Test.{uuid4()}"
        with SingleInstanceLock(name):
            with self.assertRaises(AlreadyRunningError):
                SingleInstanceLock(name).acquire()

    def test_lock_can_be_acquired_again_after_release(self) -> None:
        name = f"Local\\DesktopHealthAssistant.Test.{uuid4()}"
        with SingleInstanceLock(name):
            pass
        with SingleInstanceLock(name):
            pass


class UnifiedLauncherParserTest(unittest.TestCase):
    def test_camera_mode_is_default(self) -> None:
        self.assertFalse(build_parser().parse_args([]).no_camera)

    def test_no_camera_mode_is_explicit(self) -> None:
        self.assertTrue(build_parser().parse_args(["--no-camera"]).no_camera)


class UnifiedLauncherTrayTest(unittest.TestCase):
    @staticmethod
    def run_without_real_lock(callback) -> int:
        callback()
        return 0

    @patch("desktop_health_assistant.TrayController")
    @patch("desktop_health_assistant.neck_monitor.main")
    @patch("desktop_health_assistant.run_with_instance_lock")
    def test_camera_mode_starts_and_stops_tray(
        self,
        lock: Mock,
        camera_main: Mock,
        tray_type: Mock,
    ) -> None:
        lock.side_effect = self.run_without_real_lock

        result = main([])

        self.assertEqual(result, 0)
        tray_type.return_value.start.assert_called_once_with()
        tray_type.return_value.stop.assert_called_once_with()
        control = camera_main.call_args.kwargs["control"]
        self.assertTrue(control.camera_enabled)

    @patch("desktop_health_assistant.TrayController")
    @patch("desktop_health_assistant.reminder_service.main")
    @patch("desktop_health_assistant.run_with_instance_lock")
    def test_no_camera_mode_passes_control_and_stops_tray(
        self,
        lock: Mock,
        reminder_main: Mock,
        tray_type: Mock,
    ) -> None:
        lock.side_effect = self.run_without_real_lock

        result = main(["--no-camera"])

        self.assertEqual(result, 0)
        tray_type.return_value.start.assert_called_once_with()
        tray_type.return_value.stop.assert_called_once_with()
        control = reminder_main.call_args.kwargs["control"]
        self.assertFalse(control.camera_enabled)


if __name__ == "__main__":
    unittest.main()
