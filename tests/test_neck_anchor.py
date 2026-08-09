import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from posture_experiment import neck_anchor_mode  # noqa: E402


def landmark(x: float, y: float, z: float = 0.0, presence: float = 0.95, visibility: float = 0.98):
    return SimpleNamespace(
        x=x,
        y=y,
        z=z,
        presence=presence,
        visibility=visibility,
    )


def face_points():
    points = [landmark(0.5, 0.45) for _ in range(455)]
    points[152] = landmark(0.5, 0.72)
    points[234] = landmark(0.36, 0.45)
    points[454] = landmark(0.64, 0.45)
    return points


def pose_points(left_y: float, right_y: float):
    image = [landmark(0.5, 0.5) for _ in range(33)]
    world = [landmark(0.0, 0.0) for _ in range(33)]
    image[11] = landmark(0.20, left_y)
    image[12] = landmark(0.80, right_y)
    world[11] = landmark(0.15, 0.0, 0.02)
    world[12] = landmark(-0.15, 0.0, -0.02)
    return image, world


class NeckAnchorTest(unittest.TestCase):
    def test_uses_direct_anchor_when_shoulders_are_visible(self) -> None:
        image, world = pose_points(0.90, 0.91)
        self.assertEqual(neck_anchor_mode(face_points(), image, world)[0], "direct")

    def test_uses_estimated_anchor_below_frame(self) -> None:
        image, world = pose_points(1.18, 1.14)
        self.assertEqual(neck_anchor_mode(face_points(), image, world)[0], "estimated")

    def test_rejects_anchor_too_far_below_frame(self) -> None:
        image, world = pose_points(1.58, 1.55)
        self.assertIsNone(neck_anchor_mode(face_points(), image, world)[0])

    def test_keeps_low_visibility_anchor_for_stability_check(self) -> None:
        image, world = pose_points(1.12, 1.10)
        image[11].visibility = 0.4
        self.assertEqual(neck_anchor_mode(face_points(), image, world)[0], "estimated")


if __name__ == "__main__":
    unittest.main()
