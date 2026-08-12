import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from upper_body_contour_experiment import (  # noqa: E402
    classify_motion,
    sequence_features,
    stabilized_visibility_mode,
    upper_body_metrics,
    visibility_mode,
)


def face_points() -> list[SimpleNamespace]:
    points = [SimpleNamespace(x=0.5, y=0.3) for _ in range(455)]
    points[152] = SimpleNamespace(x=0.5, y=0.4)
    points[234] = SimpleNamespace(x=0.35, y=0.3)
    points[454] = SimpleNamespace(x=0.65, y=0.3)
    return points


def synthetic_mask(with_trapezius: bool) -> np.ndarray:
    mask = np.zeros((200, 200, 1), dtype=np.uint8)
    for y in range(80, 150):
        expansion = max(0, y - 100) if with_trapezius else 0
        left = max(0, 75 - expansion)
        right = min(199, 125 + expansion)
        mask[y, left:right + 1, 0] = 2
    return mask


def shoulder_edge_mask() -> np.ndarray:
    mask = np.zeros((200, 200, 1), dtype=np.uint8)
    mask[80:150, 75:126, 0] = 2
    for y in range(150, 200):
        expansion = (y - 150) * 2
        mask[y, max(0, 75 - expansion):min(200, 126 + expansion), 0] = 4
    return mask


class UpperBodyMetricsTest(unittest.TestCase):
    def test_uses_visible_neck_and_trapezius_contour(self) -> None:
        metrics, _ = upper_body_metrics(synthetic_mask(True), face_points())
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics["torso_width"], metrics["face_width"])
        self.assertGreater(metrics["contour_coverage"], 0.5)

    def test_accepts_straight_visible_clothed_contour(self) -> None:
        metrics, _ = upper_body_metrics(synthetic_mask(False), face_points())
        self.assertIsNotNone(metrics)

    def test_accepts_clothed_shoulder_contour(self) -> None:
        metrics, _ = upper_body_metrics(shoulder_edge_mask(), face_points())
        self.assertIsNotNone(metrics)

    def test_marks_low_head_as_too_low(self) -> None:
        self.assertEqual(visibility_mode(0.10, True), "TOO LOW")

    def test_distinguishes_partial_and_full_visibility(self) -> None:
        self.assertEqual(visibility_mode(0.30, True), "PARTIAL")
        self.assertEqual(visibility_mode(0.70, True), "FULL")

    def test_requires_contour_when_head_is_high_enough(self) -> None:
        self.assertEqual(visibility_mode(0.70, False), "CONTOUR UNSTABLE")

    def test_missing_face_is_not_reported_as_head_too_low(self) -> None:
        self.assertEqual(visibility_mode(0.0, False), "CONTOUR UNSTABLE")

    def test_visibility_hysteresis_keeps_partial_near_boundary(self) -> None:
        self.assertEqual(
            stabilized_visibility_mode("PARTIAL", 0.52, 0.90),
            "PARTIAL",
        )

    def test_visibility_hysteresis_keeps_full_near_boundary(self) -> None:
        self.assertEqual(
            stabilized_visibility_mode("FULL", 0.48, 0.90),
            "FULL",
        )

    def test_classifies_head_forward_motion(self) -> None:
        prediction = classify_motion(
            {"face_growth": 0.12, "torso_growth": 0.02, "face_y_change": 0.06}
        )
        self.assertEqual(prediction, "HEAD FORWARD")

    def test_classifies_head_forward_when_neck_area_is_occluded(self) -> None:
        prediction = classify_motion(
            {
                "face_growth": 0.14,
                "torso_growth": 0.04,
                "face_y_change": 0.01,
                "torso_area_growth": -0.44,
            }
        )
        self.assertEqual(prediction, "HEAD FORWARD")

    def test_classifies_mild_downward_head_motion(self) -> None:
        prediction = classify_motion(
            {
                "face_growth": 0.14,
                "torso_growth": 0.04,
                "face_y_change": 0.017,
                "torso_area_growth": -0.08,
            }
        )
        self.assertEqual(prediction, "HEAD FORWARD")

    def test_classifies_whole_body_forward_motion(self) -> None:
        prediction = classify_motion(
            {"face_growth": 0.25, "torso_growth": 0.15, "face_y_change": -0.04}
        )
        self.assertEqual(prediction, "WHOLE BODY FORWARD")

    def test_classifies_small_changes_as_stable(self) -> None:
        prediction = classify_motion(
            {"face_growth": 0.01, "torso_growth": -0.02, "face_y_change": 0.01}
        )
        self.assertEqual(prediction, "STABLE")

    def test_tolerates_minor_partial_contour_jitter(self) -> None:
        prediction = classify_motion(
            {"face_growth": -0.032, "torso_growth": -0.059, "face_y_change": -0.017}
        )
        self.assertEqual(prediction, "STABLE")

    def test_sequence_features_compare_start_and_end(self) -> None:
        frames = []
        for index in range(20):
            progress = index / 19
            frames.append(
                {
                    "phase": "baseline" if index < 5 else "action",
                    "metrics": {
                        "face_width": 0.25 * (1.0 + 0.12 * progress),
                        "torso_width": 0.60 * (1.0 + 0.02 * progress),
                        "face_center_y": 0.55 + 0.06 * progress,
                        "torso_center_y": 0.88,
                        "torso_area": 0.20,
                    }
                }
            )
        features = sequence_features(frames)
        self.assertIsNotNone(features)
        self.assertGreater(features["face_growth"], 0.08)
        self.assertLess(features["torso_growth"], 0.06)


if __name__ == "__main__":
    unittest.main()
