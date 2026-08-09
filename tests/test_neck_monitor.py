import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from neck_monitor import classify_posture, data_issue_message  # noqa: E402


BASELINE = {
    "median": {
        "face_width": 0.30,
        "shoulder_width": 0.50,
        "face_shoulder_ratio": 0.60,
    },
    "mad": {
        "face_width": 0.005,
        "shoulder_width": 0.010,
        "face_shoulder_ratio": 0.010,
    },
}


class NeckMonitorClassificationTest(unittest.TestCase):
    def test_normal(self) -> None:
        current = {"face_width": 0.30, "shoulder_width": 0.50, "face_shoulder_ratio": 0.60}
        self.assertEqual(classify_posture(BASELINE, current)[0], "NECK NORMAL")

    def test_head_forward(self) -> None:
        current = {"face_width": 0.35, "shoulder_width": 0.54, "face_shoulder_ratio": 0.648}
        self.assertEqual(classify_posture(BASELINE, current)[0], "NECK FORWARD")

    def test_whole_body_closer(self) -> None:
        current = {"face_width": 0.36, "shoulder_width": 0.60, "face_shoulder_ratio": 0.60}
        self.assertEqual(classify_posture(BASELINE, current)[0], "NECK NORMAL")

    def test_whole_body_closer_with_small_ratio_change_is_normal(self) -> None:
        current = {"face_width": 0.38, "shoulder_width": 0.60, "face_shoulder_ratio": 0.636}
        self.assertEqual(classify_posture(BASELINE, current)[0], "NECK NORMAL")

    def test_moving_farther_is_normal_even_if_ratio_changes(self) -> None:
        current = {"face_width": 0.285, "shoulder_width": 0.43, "face_shoulder_ratio": 0.665}
        self.assertEqual(classify_posture(BASELINE, current)[0], "NECK NORMAL")

    def test_data_issue_message_for_low_posture(self) -> None:
        self.assertEqual(
            data_issue_message("anchor_not_below_chin"),
            "TOO CLOSE OR BENT TOO LOW",
        )

    def test_data_issue_message_for_missing_face(self) -> None:
        self.assertEqual(
            data_issue_message("face_missing"),
            "FACE NOT VISIBLE - MOVE BACK OR SIT UP",
        )


if __name__ == "__main__":
    unittest.main()
