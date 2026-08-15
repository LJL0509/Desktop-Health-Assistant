import argparse

import neck_monitor
import reminder_service
from app_runtime import run_safely
from instance_lock import run_with_instance_lock
from tray_control import AppControl, TrayController, enable_dpi_awareness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desktop Health Assistant")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="run reminders and computer-use statistics without camera monitoring",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    enable_dpi_awareness()
    args = build_parser().parse_args(argv)

    def run_selected_mode() -> None:
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
