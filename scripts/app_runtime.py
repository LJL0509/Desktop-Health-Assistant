import ctypes
import logging
import os
import sys
import traceback
import winreg
from pathlib import Path

from app_paths import APP_DIRECTORY_NAME, DATA_ROOT, IS_FROZEN, LOG_DIR


AUTOSTART_NAME = "DesktopHealthAssistant"


def executable_command() -> str:
    if IS_FROZEN:
        return f'"{Path(sys.executable).resolve()}"'
    launcher = Path(__file__).resolve().parent / "desktop_health_assistant.py"
    return f'"{Path(sys.executable).resolve()}" "{launcher}"'


def autostart_command() -> str | None:
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
        ) as key:
            value, _ = winreg.QueryValueEx(key, AUTOSTART_NAME)
            return str(value)
    except FileNotFoundError:
        return None


def is_autostart_enabled() -> bool:
    return autostart_command() == executable_command()


def set_autostart_enabled(enabled: bool) -> None:
    access = winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE
    with winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0,
        access,
    ) as key:
        if enabled:
            winreg.SetValueEx(
                key,
                AUTOSTART_NAME,
                0,
                winreg.REG_SZ,
                executable_command(),
            )
        else:
            try:
                winreg.DeleteValue(key, AUTOSTART_NAME)
            except FileNotFoundError:
                pass


def open_data_directory() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    os.startfile(DATA_ROOT)


def configure_file_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "application.log"
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        encoding="utf-8",
    )
    return path


def show_error_message(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(
        None,
        message,
        APP_DIRECTORY_NAME,
        0x10,
    )


def run_safely(callback) -> int:
    log_path = configure_file_logging()
    try:
        result = callback()
        return int(result or 0)
    except Exception:
        logging.getLogger(__name__).exception("Application stopped unexpectedly")
        traceback.print_exc()
        show_error_message(
            "桌面健康助手遇到错误，已安全停止。\n\n"
            f"错误日志：{log_path}"
        )
        return 1
