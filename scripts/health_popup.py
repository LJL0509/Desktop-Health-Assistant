import ctypes
from ctypes import wintypes
import queue
import threading
import traceback
from typing import Callable


POPUP_WIDTH = 420
POPUP_MIN_HEIGHT = 180
POPUP_RIGHT_MARGIN = 24
POPUP_BOTTOM_MARGIN = 64
FIRST_REPEAT_REMINDER_SECONDS = 3 * 60.0
ONGOING_REPEAT_REMINDER_SECONDS = 10 * 60.0
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000
LWA_ALPHA = 0x00000002
HWND_TOPMOST = -1
MONITOR_DEFAULTTONEAREST = 2
SW_SHOWNOACTIVATE = 4
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
_user32 = None


def windows_user32():
    global _user32
    if _user32 is None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetParent.argtypes = [wintypes.HWND]
        user32.GetParent.restype = wintypes.HWND
        user32.GetClassNameW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        user32.GetClassNameW.restype = ctypes.c_int
        user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        user32.MonitorFromWindow.restype = wintypes.HMONITOR
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
        user32.GetMonitorInfoW.restype = wintypes.BOOL
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.ShowWindow.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.SetLayeredWindowAttributes.argtypes = [
            wintypes.HWND,
            wintypes.COLORREF,
            wintypes.BYTE,
            wintypes.DWORD,
        ]
        user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
        for name in ("GetWindowLongPtrW", "GetWindowLongW"):
            if hasattr(user32, name):
                function = getattr(user32, name)
                function.argtypes = [wintypes.HWND, ctypes.c_int]
                function.restype = ctypes.c_ssize_t
        for name in ("SetWindowLongPtrW", "SetWindowLongW"):
            if hasattr(user32, name):
                function = getattr(user32, name)
                function.argtypes = [
                    wintypes.HWND,
                    ctypes.c_int,
                    ctypes.c_ssize_t,
                ]
                function.restype = ctypes.c_ssize_t
        _user32 = user32
    return _user32


class Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", Rect),
        ("rcWork", Rect),
        ("dwFlags", ctypes.c_ulong),
    ]


def popup_geometry(
    screen_width: int,
    screen_height: int,
    requested_height: int,
    work_area: tuple[int, int, int, int] | None = None,
) -> str:
    height = max(POPUP_MIN_HEIGHT, requested_height)
    if work_area is None:
        left, top, right, bottom = 0, 0, screen_width, screen_height
        bottom_margin = POPUP_BOTTOM_MARGIN
    else:
        left, top, right, bottom = work_area
        bottom_margin = 16
    x = max(left + 16, right - POPUP_WIDTH - POPUP_RIGHT_MARGIN)
    y = max(top + 16, bottom - height - bottom_margin)
    return f"{POPUP_WIDTH}x{height}+{x}+{y}"


def popup_extended_style(existing_style: int) -> int:
    return (
        existing_style | WS_EX_TOOLWINDOW | WS_EX_LAYERED | WS_EX_NOACTIVATE
    ) & ~WS_EX_APPWINDOW


def top_level_window_handle(root, user32) -> int:
    window_handle = int(root.winfo_id())
    parent_handle = int(user32.GetParent(window_handle))
    return parent_handle or window_handle


def configure_no_activate_window(
    window_handle: int,
    user32,
) -> None:
    get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    existing_style = int(get_style(window_handle, GWL_EXSTYLE))
    set_style(
        window_handle,
        GWL_EXSTYLE,
        popup_extended_style(existing_style),
    )
    user32.SetLayeredWindowAttributes(
        window_handle,
        0,
        255,
        LWA_ALPHA,
    )


def foreground_window_context(user32) -> tuple[int, str]:
    foreground_handle = int(user32.GetForegroundWindow())
    if not foreground_handle:
        return 0, ""
    class_name = ctypes.create_unicode_buffer(256)
    if not user32.GetClassNameW(foreground_handle, class_name, len(class_name)):
        return foreground_handle, ""
    return foreground_handle, class_name.value


def monitor_work_area(window_handle: int, user32) -> tuple[int, int, int, int] | None:
    if not window_handle:
        return None
    monitor_handle = user32.MonitorFromWindow(
        window_handle,
        MONITOR_DEFAULTTONEAREST,
    )
    if not monitor_handle:
        return None
    info = MonitorInfo()
    info.cbSize = ctypes.sizeof(MonitorInfo)
    if not user32.GetMonitorInfoW(monitor_handle, ctypes.byref(info)):
        return None
    return (
        int(info.rcWork.left),
        int(info.rcWork.top),
        int(info.rcWork.right),
        int(info.rcWork.bottom),
    )


def show_window_without_activation(
    window_handle: int,
    user32,
    z_order_target: int = HWND_TOPMOST,
) -> None:
    user32.ShowWindow(window_handle, SW_SHOWNOACTIVATE)
    user32.SetWindowPos(
        window_handle,
        z_order_target,
        0,
        0,
        0,
        0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
    )


def show_tk_popup_without_activation(
    root,
    user32=None,
    z_order_target: int = HWND_TOPMOST,
) -> int:
    if user32 is None:
        user32 = windows_user32()
    window_handle = top_level_window_handle(root, user32)
    configure_no_activate_window(window_handle, user32)
    show_window_without_activation(window_handle, user32, z_order_target)
    return window_handle


def keep_tk_popup_visible(window_handle: int, z_order_target: int) -> None:
    show_window_without_activation(
        window_handle,
        windows_user32(),
        z_order_target,
    )


class HealthPopupHost:
    def __init__(self) -> None:
        self.commands: queue.Queue[tuple[str, str]] = queue.Queue()
        self.ready = threading.Event()
        self.popup_shown = threading.Event()
        self.error: BaseException | None = None
        self.last_popup_error: BaseException | None = None
        self.last_window_handle: int | None = None
        self.root = None
        self.thread = threading.Thread(
            target=self._run,
            name="health-popup-host",
            daemon=True,
        )
        self.thread.start()
        if not self.ready.wait(5.0):
            raise RuntimeError("Health popup host did not start")
        if self.error is not None:
            raise RuntimeError("Health popup host failed to start") from self.error

    def show(self, title: str, message: str) -> None:
        self.popup_shown.clear()
        self.last_popup_error = None
        self.commands.put((title, message))

    def wait_until_shown(self, timeout: float = 5.0) -> bool:
        return self.popup_shown.wait(timeout)

    def _run(self) -> None:
        try:
            import tkinter as tk

            self.root = tk.Tk(className="DesktopHealthAssistantAlertHost")
            self.root.withdraw()
            self.root.after(20, self._process_commands)
            self.ready.set()
            self.root.mainloop()
        except BaseException as error:
            self.error = error
            self.ready.set()

    def _process_commands(self) -> None:
        if self.root is None:
            return
        while True:
            try:
                title, message = self.commands.get_nowait()
            except queue.Empty:
                break
            try:
                self._create_popup(title, message)
            except BaseException as error:
                self.last_popup_error = error
                traceback.print_exc()
        self.root.after(50, self._process_commands)

    def _create_popup(self, title: str, message: str) -> None:
        import tkinter as tk

        user32 = windows_user32()
        foreground_handle, _ = foreground_window_context(user32)
        work_area = monitor_work_area(foreground_handle, user32)

        window = tk.Toplevel(self.root, class_="DesktopHealthAssistantAlert")
        window.withdraw()
        window.title(title)
        window.configure(background="#171b1f")
        window.resizable(False, False)
        window.overrideredirect(True)

        border = tk.Frame(window, background="#ef5b6c", padx=2, pady=2)
        border.pack(fill="both", expand=True)
        panel = tk.Frame(border, background="#20262b")
        panel.pack(fill="both", expand=True)

        tk.Label(
            panel,
            text=title,
            font=("Microsoft YaHei UI", 14, "bold"),
            foreground="#ffffff",
            background="#20262b",
            anchor="w",
        ).pack(fill="x", padx=18, pady=(15, 5))
        tk.Label(
            panel,
            text=message,
            font=("Microsoft YaHei UI", 10),
            foreground="#d9dee2",
            background="#20262b",
            justify="left",
            wraplength=380,
            anchor="w",
        ).pack(fill="x", padx=18)
        tk.Button(
            panel,
            text="知道了",
            command=window.destroy,
            font=("Microsoft YaHei UI", 9),
            foreground="#ffffff",
            background="#c84455",
            activeforeground="#ffffff",
            activebackground="#a93645",
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=5,
            cursor="hand2",
        ).pack(anchor="e", padx=18, pady=(10, 12))

        window.update_idletasks()
        window.geometry(
            popup_geometry(
                window.winfo_screenwidth(),
                window.winfo_screenheight(),
                border.winfo_reqheight(),
                work_area,
            )
        )
        window.update_idletasks()
        window_handle = show_tk_popup_without_activation(
            window,
            user32,
        )
        self.last_window_handle = window_handle
        self.popup_shown.set()
        window.after(15000, window.destroy)
        window.after(
            50,
            lambda: keep_tk_popup_visible(window_handle, HWND_TOPMOST),
        )


_popup_host: HealthPopupHost | None = None
_popup_host_lock = threading.Lock()


def initialize_health_popup_host() -> HealthPopupHost:
    global _popup_host
    with _popup_host_lock:
        if _popup_host is None:
            _popup_host = HealthPopupHost()
        return _popup_host


def show_health_popup(title: str, message: str) -> None:
    initialize_health_popup_host().show(title, message)


def show_context_aware_health_alert(title: str, message: str) -> str:
    show_health_popup(title, message)
    return "custom_popup"


class HealthPopupNotifier:
    def __init__(
        self,
        popup: Callable[[str, str], None] | None = None,
    ) -> None:
        if popup is None:
            initialize_health_popup_host()
            self.popup = show_context_aware_health_alert
        else:
            self.popup = popup

    def show(self, title: str, message: str) -> None:
        threading.Thread(
            target=self.popup,
            args=(title, message),
            daemon=True,
        ).start()
