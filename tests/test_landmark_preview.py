import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from landmark_preview import angle_degrees, assess_shoulders, shoulders_are_valid  # noqa: E402


def landmark(x: float, y: float, visibility: float = 0.9, presence: float = 0.9):
    return SimpleNamespace(x=x, y=y, visibility=visibility, presence=presence)


def valid_face():
    points = [landmark(0.5, 0.4) for _ in range(455)]
    points[152] = landmark(0.5, 0.55)
    points[234] = landmark(0.35, 0.42)
    points[454] = landmark(0.65, 0.42)
    return points


class ShoulderValidationTest(unittest.TestCase):
    def test_accepts_visible_shoulders_below_face(self) -> None:
        pose = [landmark(0.5, 0.5) for _ in range(33)]
        pose[11] = landmark(0.2, 0.78)
        pose[12] = landmark(0.8, 0.80)
        self.assertTrue(shoulders_are_valid(pose, valid_face()))

    def test_rejects_shoulders_outside_frame(self) -> None:
        pose = [landmark(0.5, 0.5) for _ in range(33)]
        pose[11] = landmark(0.2, 1.08)
        pose[12] = landmark(0.8, 1.10)
        self.assertFalse(shoulders_are_valid(pose, valid_face()))

    def test_accepts_shoulders_near_bottom_edge(self) -> None:
        pose = [landmark(0.5, 0.5) for _ in range(33)]
        pose[11] = landmark(0.22, 0.994)
        pose[12] = landmark(0.78, 0.987)
        self.assertTrue(shoulders_are_valid(pose, valid_face()))

    def test_marks_slightly_outside_shoulders_as_partial(self) -> None:
        pose = [landmark(0.5, 0.5) for _ in range(33)]
        pose[11] = landmark(0.81, 1.052, visibility=0.99, presence=0.91)
        pose[12] = landmark(0.24, 1.020, visibility=0.99, presence=0.97)
        self.assertEqual(assess_shoulders(pose, valid_face())[0], "partial")

    def test_rejects_shoulders_too_far_outside(self) -> None:
        pose = [landmark(0.5, 0.5) for _ in range(33)]
        pose[11] = landmark(0.2, 1.12)
        pose[12] = landmark(0.8, 1.11)
        self.assertEqual(assess_shoulders(pose, valid_face())[0], "missing")

    def test_rejects_inferred_shoulders_on_face(self) -> None:
        pose = [landmark(0.5, 0.5) for _ in range(33)]
        pose[11] = landmark(0.34, 0.48)
        pose[12] = landmark(0.66, 0.49)
        self.assertFalse(shoulders_are_valid(pose, valid_face()))

    def test_normalizes_reversed_shoulder_angle(self) -> None:
        self.assertAlmostEqual(
            angle_degrees(landmark(0.8, 0.5), landmark(0.2, 0.52)),
            -1.91,
            places=1,
        )


if __name__ == "__main__":
    unittest.main()
