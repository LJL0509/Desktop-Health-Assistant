import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from health_popup import (  # noqa: E402
    HealthPopupNotifier,
    POPUP_MIN_HEIGHT,
    popup_geometry,
)


class HealthPopupGeometryTest(unittest.TestCase):
    def test_expands_to_fit_high_dpi_content(self) -> None:
        self.assertEqual(
            popup_geometry(1920, POPUP_MIN_HEIGHT + 42),
            "420x222+1476+36",
        )

    def test_keeps_minimum_height_for_short_content(self) -> None:
        self.assertEqual(
            popup_geometry(1920, 100),
            f"420x{POPUP_MIN_HEIGHT}+1476+36",
        )


class HealthPopupNotifierTest(unittest.TestCase):
    @patch("health_popup.threading.Thread")
    def test_starts_popup_without_blocking_monitoring(self, thread_type: Mock) -> None:
        popup = Mock()
        notifier = HealthPopupNotifier(popup)

        notifier.show("健康提醒", "请休息一下")

        thread_type.assert_called_once_with(
            target=popup,
            args=("健康提醒", "请休息一下"),
            daemon=True,
        )
        thread_type.return_value.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
