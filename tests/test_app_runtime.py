import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import app_runtime  # noqa: E402


class AppRuntimeTest(unittest.TestCase):
    def test_source_autostart_command_uses_python_and_launcher(self) -> None:
        command = app_runtime.executable_command()
        self.assertIn("python", command.lower())
        self.assertIn("desktop_health_assistant.py", command)

    @patch("app_runtime.configure_file_logging")
    def test_safe_runner_returns_callback_result(self, configure: Mock) -> None:
        configure.return_value = Path("application.log")
        self.assertEqual(app_runtime.run_safely(lambda: 7), 7)

    @patch("app_runtime.show_error_message")
    @patch("app_runtime.configure_file_logging")
    def test_safe_runner_reports_unhandled_error(
        self,
        configure: Mock,
        show_error: Mock,
    ) -> None:
        configure.return_value = Path("application.log")

        def fail() -> None:
            raise RuntimeError("boom")

        self.assertEqual(app_runtime.run_safely(fail), 1)
        show_error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
