import ctypes
import os
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

from daily_report import REPORT_DIR


WINDOW_HIDE = 0
WINDOW_SHOW = 5


@dataclass(frozen=True)
class ControlSnapshot:
    camera_paused: bool
    window_visible: bool
    exit_requested: bool


class AppControl:
    def __init__(self, camera_enabled: bool) -> None:
        self.camera_enabled = camera_enabled
        self._camera_paused = False
        self._window_visible = True
        self._exit_requested = False
        self._lock = threading.Lock()

    def snapshot(self) -> ControlSnapshot:
        with self._lock:
            return ControlSnapshot(
                camera_paused=self._camera_paused,
                window_visible=self._window_visible,
                exit_requested=self._exit_requested,
            )

    def toggle_camera_paused(self) -> None:
        if not self.camera_enabled:
            return
        with self._lock:
            self._camera_paused = not self._camera_paused

    def set_camera_paused(self, paused: bool) -> None:
        with self._lock:
            self._camera_paused = paused

    def toggle_window_visible(self) -> None:
        if not self.camera_enabled:
            return
        with self._lock:
            self._window_visible = not self._window_visible

    def request_exit(self) -> None:
        with self._lock:
            self._exit_requested = True


def latest_report(report_dir: Path = REPORT_DIR) -> Path | None:
    reports = []
    if report_dir.exists():
        for path in report_dir.glob("*.md"):
            try:
                date.fromisoformat(path.stem)
            except ValueError:
                continue
            reports.append(path)
    return max(reports, key=lambda path: path.stem, default=None)


def open_latest_report(report_dir: Path = REPORT_DIR) -> bool:
    report = latest_report(report_dir)
    if report is None:
        return False
    os.startfile(report)
    return True


def set_native_window_visible(title: str, visible: bool) -> bool:
    if not hasattr(ctypes, "windll"):
        return False
    window = ctypes.windll.user32.FindWindowW(None, title)
    if not window:
        return False
    ctypes.windll.user32.ShowWindow(
        window,
        WINDOW_SHOW if visible else WINDOW_HIDE,
    )
    return True


def build_icon_image(size: int = 64) -> Image.Image:
    image = Image.new("RGBA", (size, size), (30, 35, 40, 255))
    draw = ImageDraw.Draw(image)
    margin = size // 7
    draw.rounded_rectangle(
        (margin, margin, size - margin, size - margin),
        radius=size // 8,
        fill=(50, 170, 125, 255),
    )
    center = size // 2
    stroke = max(3, size // 11)
    draw.line((center, margin * 2, center, size - margin * 2), fill="white", width=stroke)
    draw.line((margin * 2, center, size - margin * 2, center), fill="white", width=stroke)
    return image


class TrayController:
    def __init__(
        self,
        control: AppControl,
        report_dir: Path = REPORT_DIR,
    ) -> None:
        self.control = control
        self.report_dir = report_dir
        self.icon = Icon(
            "desktop-health-assistant",
            build_icon_image(),
            "桌面健康助手",
            self._build_menu(),
        )

    def _build_menu(self) -> Menu:
        return Menu(
            MenuItem(
                lambda _item: (
                    "隐藏监测窗口"
                    if self.control.snapshot().window_visible
                    else "显示监测窗口"
                ),
                self._toggle_window,
                visible=lambda _item: self.control.camera_enabled,
            ),
            MenuItem(
                lambda _item: (
                    "继续摄像头监测"
                    if self.control.snapshot().camera_paused
                    else "暂停摄像头监测"
                ),
                self._toggle_camera,
                visible=lambda _item: self.control.camera_enabled,
            ),
            MenuItem("打开最新报告", self._open_report),
            Menu.SEPARATOR,
            MenuItem("退出", self._exit),
        )

    def start(self) -> None:
        self.icon.run_detached()

    def stop(self) -> None:
        self.icon.stop()

    def _toggle_window(self, icon, _item) -> None:
        self.control.toggle_window_visible()
        icon.update_menu()

    def _toggle_camera(self, icon, _item) -> None:
        self.control.toggle_camera_paused()
        icon.update_menu()

    def _open_report(self, _icon, _item) -> None:
        open_latest_report(self.report_dir)

    def _exit(self, icon, _item) -> None:
        self.control.request_exit()
        icon.stop()
