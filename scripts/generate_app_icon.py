import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from tray_control import build_icon_image  # noqa: E402


def main() -> None:
    asset_dir = ROOT / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    image = build_icon_image(256)
    image.save(asset_dir / "desktop-health-assistant.png")
    image.save(
        asset_dir / "desktop-health-assistant.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


if __name__ == "__main__":
    main()
