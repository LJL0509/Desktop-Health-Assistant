import argparse
import time

import neck_monitor
import reminder_service
from app_runtime import run_safely
from health_popup import (
    initialize_health_popup_host,
    show_context_aware_health_alert,
)
from instance_lock import run_with_instance_lock
from tray_control import AppControl, TrayController, enable_dpi_awareness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desktop Health Assistant")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="run reminders and computer-use statistics without camera monitoring",
    )
    parser.add_argument(
        "--test-health-popup",
        action="store_true",
        help="show one health popup without starting camera monitoring",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    enable_dpi_awareness()
    args = build_parser().parse_args(argv)

    if args.test_health_popup:
        host = initialize_health_popup_host()
        print(
            "Health popup host is ready. Switch to the fullscreen window; "
            "the reminder will appear in 5 seconds.",
            flush=True,
        )
        time.sleep(5.0)
        delivery = show_context_aware_health_alert(
            "健康提醒测试",
            "这是一次弹窗显示测试。它不应让全屏窗口退出，也不应抢走键盘焦点。",
        )
        if delivery == "custom_popup" and not host.wait_until_shown(5.0):
            if host.last_popup_error is not None:
                raise RuntimeError("Health popup could not be displayed") from host.last_popup_error
            raise RuntimeError("Health popup was not displayed within 5 seconds")
        print(
            f"Health alert delivered via {delivery} "
            f"(HWND={host.last_window_handle}).",
            flush=True,
        )
        time.sleep(16.0)
        return 0

    def run_selected_mode() -> None:
        if not args.no_camera:
            initialize_health_popup_host()
        control = AppControl(camera_enabled=not args.no_camera)
        tray = TrayController(control)
        tray.start()
        try:
            if args.no_camera:
                reminder_service.main([], control=control)
            else:
                neck_monitor.main(control=control)
        finally:
            tray.stop()

    return run_with_instance_lock(run_selected_mode)


if __name__ == "__main__":
    raise SystemExit(run_safely(main))
