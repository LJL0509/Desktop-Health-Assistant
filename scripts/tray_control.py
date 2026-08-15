import ctypes
import math
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

from app_runtime import (
    is_autostart_enabled,
    open_data_directory,
    set_autostart_enabled,
)
from daily_report import REPORT_DIR


WINDOW_HIDE = 0
WINDOW_SHOW = 5
WINDOW_RESTORE = 9


@dataclass(frozen=True)
class ControlSnapshot:
    camera_paused: bool
    window_visible: bool
    exit_requested: bool
    hydration_started_at: datetime | None
    last_water_at: datetime | None
    hydration_interval_seconds: float


class AppControl:
    def __init__(self, camera_enabled: bool) -> None:
        self.camera_enabled = camera_enabled
        self._camera_paused = False
        self._window_visible = True
        self._exit_requested = False
        self._hydration_started_at: datetime | None = None
        self._last_water_at: datetime | None = None
        self._hydration_interval_seconds = 60.0 * 60.0
        self._water_requests: list[int] = []
        self._lock = threading.Lock()

    def snapshot(self) -> ControlSnapshot:
        with self._lock:
            return ControlSnapshot(
                camera_paused=self._camera_paused,
                window_visible=self._window_visible,
                exit_requested=self._exit_requested,
                hydration_started_at=self._hydration_started_at,
                last_water_at=self._last_water_at,
                hydration_interval_seconds=self._hydration_interval_seconds,
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

    def update_hydration_status(
        self,
        started_at: datetime,
        last_water_at: datetime | None,
        interval_seconds: float,
    ) -> None:
        with self._lock:
            self._hydration_started_at = started_at
            self._last_water_at = last_water_at
            self._hydration_interval_seconds = interval_seconds

    def request_water_record(self, amount_ml: int = 250) -> None:
        with self._lock:
            self._water_requests.append(amount_ml)

    def consume_water_requests(self) -> list[int]:
        with self._lock:
            requests = self._water_requests
            self._water_requests = []
            return requests


def hydration_elapsed_seconds(
    snapshot: ControlSnapshot,
    now: datetime | None = None,
) -> float | None:
    anchor = snapshot.last_water_at or snapshot.hydration_started_at
    if anchor is None:
        return None
    current = now or datetime.now().astimezone()
    return max(0.0, (current - anchor).total_seconds())


def format_elapsed_time(seconds: float | None) -> str:
    if seconds is None:
        return "正在初始化"
    total_minutes = int(seconds // 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分钟"
    return f"{minutes}分钟"


def enable_dpi_awareness() -> None:
    if not hasattr(ctypes, "windll"):
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


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
    if visible:
        ctypes.windll.user32.ShowWindow(window, WINDOW_RESTORE)
        ctypes.windll.user32.SetForegroundWindow(window)
    else:
        ctypes.windll.user32.ShowWindow(window, WINDOW_HIDE)
    return True


def build_icon_image(size: int = 64) -> Image.Image:
    scale = 4
    canvas_size = size * scale
    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    points = []
    for index in range(181):
        angle = math.pi * 2 * index / 180
        x = 16 * math.sin(angle) ** 3
        y = (
            13 * math.cos(angle)
            - 5 * math.cos(2 * angle)
            - 2 * math.cos(3 * angle)
            - math.cos(4 * angle)
        )
        points.append(
            (
                canvas_size * (0.5 + x / 38),
                canvas_size * (0.50 - y / 38),
            )
        )
    draw.polygon(points, fill=(235, 76, 100, 255))
    draw.line(
        (
            canvas_size * 0.22,
            canvas_size * 0.52,
            canvas_size * 0.39,
            canvas_size * 0.52,
            canvas_size * 0.46,
            canvas_size * 0.39,
            canvas_size * 0.55,
            canvas_size * 0.64,
            canvas_size * 0.63,
            canvas_size * 0.48,
            canvas_size * 0.78,
            canvas_size * 0.48,
        ),
        fill=(255, 255, 255, 245),
        width=max(4, canvas_size // 25),
        joint="curve",
    )
    return image.resize((size, size), Image.Resampling.LANCZOS)


class TrayController:
    def __init__(
        self,
        control: AppControl,
        report_dir: Path = REPORT_DIR,
    ) -> None:
        self.control = control
        self.report_dir = report_dir
        self._stop_refresh = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        self._last_status_text = ""
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
                default=True,
                visible=lambda _item: self.control.camera_enabled,
            ),
            MenuItem(
                lambda _item: self._hydration_status_text(),
                None,
                enabled=False,
            ),
            MenuItem("已补水 250 ml", self._record_water),
            Menu.SEPARATOR,
            MenuItem("打开最新报告", self._open_report),
            MenuItem(
                "更多",
                Menu(
                    MenuItem(
                        "开机自动启动",
                        self._toggle_autostart,
                        checked=lambda _item: is_autostart_enabled(),
                    ),
                    MenuItem("打开数据目录", self._open_data_directory),
                    Menu.SEPARATOR,
                    MenuItem(
                        lambda _item: (
                            "继续摄像头监测"
                            if self.control.snapshot().camera_paused
                            else "暂停并释放摄像头"
                        ),
                        self._toggle_camera,
                    ),
                ),
                visible=lambda _item: self.control.camera_enabled,
            ),
            Menu.SEPARATOR,
            MenuItem("退出", self._exit),
        )

    def start(self) -> None:
        self.icon.run_detached()
        self._refresh_thread = threading.Thread(
            target=self._refresh_menu,
            daemon=True,
        )
        self._refresh_thread.start()

    def stop(self) -> None:
        self._stop_refresh.set()
        self.icon.stop()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=2.0)

    def _refresh_menu(self) -> None:
        while not self._stop_refresh.wait(1.0):
            status_text = self._hydration_status_text()
            if status_text != self._last_status_text:
                self._last_status_text = status_text
                self.icon.update_menu()

    def _hydration_status_text(self) -> str:
        snapshot = self.control.snapshot()
        elapsed = hydration_elapsed_seconds(snapshot)
        prefix = "距上次补水" if snapshot.last_water_at else "已连续使用"
        return f"{prefix} {format_elapsed_time(elapsed)}"

    def _toggle_window(self, icon, _item) -> None:
        self.control.toggle_window_visible()
        icon.update_menu()

    def _toggle_camera(self, icon, _item) -> None:
        self.control.toggle_camera_paused()
        icon.update_menu()

    def _record_water(self, icon, _item) -> None:
        self.control.request_water_record()
        icon.update_menu()

    def _open_report(self, _icon, _item) -> None:
        open_latest_report(self.report_dir)

    def _toggle_autostart(self, icon, _item) -> None:
        set_autostart_enabled(not is_autostart_enabled())
        icon.update_menu()

    def _open_data_directory(self, _icon, _item) -> None:
        open_data_directory()

    def _exit(self, icon, _item) -> None:
        self.control.request_exit()
        icon.stop()
