import os
import sys
from pathlib import Path


APP_DIRECTORY_NAME = "Desktop Health Assistant"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
IS_FROZEN = bool(getattr(sys, "frozen", False))
RESOURCE_ROOT = (
    Path(getattr(sys, "_MEIPASS")).resolve()
    if IS_FROZEN
    else SOURCE_ROOT
)


def user_data_root() -> Path:
    override = os.environ.get("DESKTOP_HEALTH_ASSISTANT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if not IS_FROZEN:
        return SOURCE_ROOT / "data"
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / APP_DIRECTORY_NAME / "data"


DATA_ROOT = user_data_root()
MODEL_ROOT = RESOURCE_ROOT / "models"
LOG_DIR = DATA_ROOT / "logs"


def resource_path(*parts: str) -> Path:
    return RESOURCE_ROOT.joinpath(*parts)


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def app_version() -> str:
    try:
        return resource_path("VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"
