import ctypes
import sys
from ctypes import wintypes


LOCK_NAME = "Local\\DesktopHealthAssistant.SingleInstance"
ERROR_ALREADY_EXISTS = 183


class AlreadyRunningError(RuntimeError):
    pass


class SingleInstanceLock:
    def __init__(self, name: str = LOCK_NAME) -> None:
        self.name = name
        self.handle = None

    def acquire(self) -> None:
        if self.handle is not None:
            return
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("Single-instance locking is only available on Windows")

        kernel32 = ctypes.windll.kernel32
        create_mutex = kernel32.CreateMutexW
        create_mutex.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        create_mutex.restype = wintypes.HANDLE
        handle = create_mutex(None, False, self.name)
        if not handle:
            raise ctypes.WinError()
        if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            raise AlreadyRunningError(
                "Desktop Health Assistant is already running."
            )
        self.handle = handle

    def release(self) -> None:
        if self.handle is None:
            return
        ctypes.windll.kernel32.CloseHandle(self.handle)
        self.handle = None

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def run_with_instance_lock(callback) -> int:
    try:
        with SingleInstanceLock():
            callback()
    except AlreadyRunningError:
        message = "桌面健康助手已经在运行，请检查系统托盘。"
        print(message)
        if getattr(sys, "frozen", False):
            ctypes.windll.user32.MessageBoxW(None, message, "桌面健康助手", 0x40)
        return 2
    return 0
