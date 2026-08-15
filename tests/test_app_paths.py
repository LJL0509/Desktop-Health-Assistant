import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from app_paths import DATA_ROOT, IS_FROZEN, app_version  # noqa: E402


class AppPathsTest(unittest.TestCase):
    def test_source_mode_keeps_existing_project_data(self) -> None:
        self.assertFalse(IS_FROZEN)
        self.assertEqual(DATA_ROOT, ROOT / "data")

    def test_reads_version_from_resource_root(self) -> None:
        self.assertEqual(app_version(), (ROOT / "VERSION").read_text().strip())

    def test_data_directory_can_be_overridden_for_packaged_smoke_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["DESKTOP_HEALTH_ASSISTANT_DATA_DIR"] = directory
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from app_paths import DATA_ROOT; print(DATA_ROOT)",
                ],
                cwd=SCRIPTS,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(Path(result.stdout.strip()), Path(directory))


if __name__ == "__main__":
    unittest.main()
