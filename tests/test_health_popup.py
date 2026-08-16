import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from health_popup import (  # noqa: E402
    configure_no_activate_window,
    GWL_EXSTYLE,
    HealthPopupNotifier,
    POPUP_MIN_HEIGHT,
    popup_geometry,
    popup_extended_style,
    show_context_aware_health_alert,
    show_window_without_activation,
    show_tk_popup_without_activation,
    SWP_NOACTIVATE,
    SWP_NOMOVE,
    SWP_NOSIZE,
    SWP_SHOWWINDOW,
    SW_SHOWNOACTIVATE,
    WS_EX_APPWINDOW,
    WS_EX_LAYERED,
    WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW,
)


class HealthPopupGeometryTest(unittest.TestCase):
    def test_expands_to_fit_high_dpi_content(self) -> None:
        self.assertEqual(
            popup_geometry(1920, 1080, POPUP_MIN_HEIGHT + 42),
            "420x222+1476+794",
        )

    def test_keeps_minimum_height_for_short_content(self) -> None:
        self.assertEqual(
            popup_geometry(1920, 1080, 100),
            f"420x{POPUP_MIN_HEIGHT}+1476+836",
        )

    def test_keeps_popup_on_screen_when_height_is_large(self) -> None:
        self.assertEqual(
            popup_geometry(1280, 720, 700),
            "420x700+836+16",
        )

    def test_uses_work_area_to_stay_above_scaled_taskbar(self) -> None:
        self.assertEqual(
            popup_geometry(2560, 1600, 180, (0, 0, 2560, 1528)),
            "420x180+2116+1332",
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

    @patch("health_popup.show_health_popup")
    def test_all_contexts_use_layered_custom_popup(
        self,
        custom_popup: Mock,
    ) -> None:
        delivery = show_context_aware_health_alert("健康提醒", "请休息一下")

        self.assertEqual(delivery, "custom_popup")
        custom_popup.assert_called_once_with("健康提醒", "请休息一下")

class HealthPopupHostStateTest(unittest.TestCase):
    def test_wait_until_shown_reports_win32_show_completion(self) -> None:
        from health_popup import HealthPopupHost

        host = HealthPopupHost.__new__(HealthPopupHost)
        host.popup_shown = __import__("threading").Event()

        self.assertFalse(host.wait_until_shown(0.0))
        host.popup_shown.set()
        self.assertTrue(host.wait_until_shown(0.0))


class NoActivatePopupTest(unittest.TestCase):
    def test_extended_style_prevents_focus_and_alt_tab_entry(self) -> None:
        existing_style = WS_EX_APPWINDOW
        self.assertEqual(
            popup_extended_style(existing_style),
            WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED,
        )

    def test_configures_window_before_showing_it(self) -> None:
        user32 = Mock()
        user32.GetWindowLongPtrW.return_value = 0x00040000

        configure_no_activate_window(1234, user32)

        user32.GetWindowLongPtrW.assert_called_once_with(1234, GWL_EXSTYLE)
        user32.SetWindowLongPtrW.assert_called_once_with(
            1234,
            GWL_EXSTYLE,
            WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_LAYERED,
        )
        user32.SetLayeredWindowAttributes.assert_called_once_with(
            1234,
            0,
            255,
            2,
        )

    def test_shows_topmost_without_activating_window(self) -> None:
        user32 = Mock()

        show_window_without_activation(1234, user32)

        user32.ShowWindow.assert_called_once_with(1234, SW_SHOWNOACTIVATE)
        user32.SetWindowPos.assert_called_once_with(
            1234,
            -1,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def test_can_place_popup_directly_above_browser(self) -> None:
        user32 = Mock()

        show_window_without_activation(1234, user32, 5678)

        user32.SetWindowPos.assert_called_once_with(
            1234,
            5678,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def test_tk_window_is_shown_only_through_no_activate_api(self) -> None:
        root = Mock()
        root.winfo_id.return_value = 1234
        user32 = Mock()
        user32.GetParent.return_value = 5678
        user32.GetWindowLongPtrW.return_value = 0

        window_handle = show_tk_popup_without_activation(root, user32)

        self.assertEqual(window_handle, 5678)
        root.deiconify.assert_not_called()
        user32.ShowWindow.assert_called_once_with(5678, SW_SHOWNOACTIVATE)

if __name__ == "__main__":
    unittest.main()
