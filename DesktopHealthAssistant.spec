from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs


ROOT = Path(SPECPATH).resolve()

hidden_imports = [
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "mediapipe.tasks.python.core.base_options",
    "mediapipe.tasks.c",
    "windows_toasts",
    "winrt.windows.data.xml.dom",
    "winrt.windows.foundation",
    "winrt.windows.ui.notifications",
]

a = Analysis(
    [str(ROOT / "scripts" / "desktop_health_assistant.py")],
    pathex=[str(ROOT / "scripts")],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=[
        (str(ROOT / "models" / "face_landmarker.task"), "models"),
        (str(ROOT / "models" / "selfie_multiclass_256x256.tflite"), "models"),
        (str(ROOT / "VERSION"), "."),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["jupyter", "notebook", "pandas", "scipy"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DesktopHealthAssistant",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "desktop-health-assistant.ico"),
    version=str(ROOT / "packaging" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Desktop Health Assistant",
)
