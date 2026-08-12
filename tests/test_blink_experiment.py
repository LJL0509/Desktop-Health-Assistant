import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from blink_experiment import (  # noqa: E402
    BLINK_RATE_RECOVERY_PER_MINUTE,
    BlinkDetector,
    BlinkRateMonitor,
    LOW_BLINK_RATE_PER_MINUTE,
    build_open_eye_baseline,
    normalized_openness,
)
from landmark_preview import next_video_timestamp_ms  # noqa: E402


def sample(left: float = 0.28, right: float = 0.27) -> dict[str, float]:
    return {
        "left_eye_ratio": left,
        "right_eye_ratio": right,
        "left_blink_score": 0.02,
        "right_blink_score": 0.03,
    }


class BlinkCalibrationTest(unittest.TestCase):
    def test_uses_upper_percentile_to_ignore_brief_blinks(self) -> None:
        samples = [sample()] * 80 + [sample(0.08, 0.08)] * 20
        baseline, message = build_open_eye_baseline(samples)
        self.assertEqual(message, "ok")
        self.assertAlmostEqual(baseline["left_open_ratio"], 0.28)
        self.assertAlmostEqual(baseline["right_open_ratio"], 0.27)

    def test_rejects_too_few_samples(self) -> None:
        baseline, message = build_open_eye_baseline([sample()] * 20)
        self.assertIsNone(baseline)
        self.assertEqual(message, "not enough valid eye samples")

    def test_integrated_monitor_can_use_thirty_valid_samples(self) -> None:
        baseline, message = build_open_eye_baseline(
            [sample()] * 30,
            minimum_samples=30,
        )
        self.assertEqual(message, "ok")
        self.assertIsNotNone(baseline)

    def test_normalizes_each_eye_against_its_own_baseline(self) -> None:
        baseline = {"left_open_ratio": 0.28, "right_open_ratio": 0.25}
        left, right = normalized_openness(sample(0.14, 0.20), baseline)
        self.assertAlmostEqual(left, 0.5)
        self.assertAlmostEqual(right, 0.8)

    def test_video_timestamp_is_strictly_increasing_within_same_millisecond(self) -> None:
        first = next_video_timestamp_ms(2.0181, 2017)
        second = next_video_timestamp_ms(2.0187, first)
        self.assertEqual(first, 2018)
        self.assertEqual(second, 2019)

    def test_video_timestamp_handles_clock_rounding_without_going_back(self) -> None:
        self.assertEqual(next_video_timestamp_ms(2.0179, 2018), 2019)


class BlinkDetectorTest(unittest.TestCase):
    def test_counts_bilateral_short_closure_as_one_blink(self) -> None:
        detector = BlinkDetector()
        detector.update(0.0, 1.0, 1.0)
        detector.update(1.0, 0.3, 0.3)
        detector.update(1.06, 0.3, 0.3)
        events = detector.update(1.18, 1.0, 1.0)
        self.assertEqual([event.kind for event in events], ["blink"])

    def test_does_not_count_single_eye_wink(self) -> None:
        detector = BlinkDetector()
        detector.update(1.0, 0.4, 1.0)
        detector.update(1.1, 0.4, 1.0)
        self.assertEqual(detector.update(1.2, 1.0, 1.0), [])

    def test_half_closed_eyes_do_not_start_a_closure(self) -> None:
        detector = BlinkDetector()
        detector.update(1.0, 0.55, 0.56, 0.36, 0.34)
        detector.update(2.0, 0.54, 0.57, 0.35, 0.36)
        self.assertEqual(detector.update(3.0, 1.0, 1.0, 0.1, 0.1), [])
        self.assertEqual(detector.state, "OPEN")


class BlinkRateMonitorTest(unittest.TestCase):
    def test_product_defaults_use_fifteen_blinks_per_minute(self) -> None:
        self.assertEqual(LOW_BLINK_RATE_PER_MINUTE, 15.0)
        self.assertEqual(BLINK_RATE_RECOVERY_PER_MINUTE, 18.0)

    def test_alerts_only_after_full_valid_observation_window(self) -> None:
        monitor = BlinkRateMonitor(window_seconds=10, low_rate_per_minute=6)
        monitor.update(0, True)
        for second in range(1, 10):
            self.assertEqual(monitor.update(second, True), [])
        events = monitor.update(10, True)
        self.assertEqual([event["event"] for event in events], ["low_blink_rate_alert"])

    def test_missing_face_does_not_advance_valid_observation(self) -> None:
        monitor = BlinkRateMonitor(window_seconds=10, low_rate_per_minute=6)
        monitor.update(0, True)
        for second in range(1, 6):
            monitor.update(second, True)
        monitor.update(105, False)
        self.assertEqual(monitor.effective_time, 5.0)
        self.assertEqual(monitor.update(114, True), [])

    def test_enough_blinks_do_not_alert(self) -> None:
        monitor = BlinkRateMonitor(
            window_seconds=10,
            low_rate_per_minute=6,
            recovery_rate_per_minute=8,
        )
        monitor.update(0, True)
        for second in range(1, 10):
            monitor.update(second, True, blink=second == 5)
        self.assertEqual(monitor.update(10, True), [])

    def test_recovery_uses_higher_rate_hysteresis(self) -> None:
        monitor = BlinkRateMonitor(
            window_seconds=10,
            low_rate_per_minute=6,
            recovery_rate_per_minute=8,
        )
        monitor.update(0, True)
        for second in range(1, 11):
            monitor.update(second, True)
        self.assertTrue(monitor.alerted)
        self.assertEqual(monitor.update(11, True, blink=True), [])
        events = monitor.update(12, True, blink=True)
        self.assertEqual(
            [event["event"] for event in events],
            ["low_blink_rate_recovered"],
        )


class BlinkDetectorRecoveryTest(unittest.TestCase):

    def test_blendshape_can_confirm_a_shallow_blink(self) -> None:
        detector = BlinkDetector()
        detector.update(1.0, 0.58, 0.60, 0.68, 0.66)
        detector.update(1.08, 0.50, 0.52, 0.72, 0.70)
        events = detector.update(1.2, 1.0, 1.0, 0.1, 0.1)
        self.assertEqual([event.kind for event in events], ["blink"])

    def test_low_blink_score_ends_closure_before_full_geometric_recovery(self) -> None:
        detector = BlinkDetector(long_closure_seconds=0.8)
        detector.update(1.0, 0.2, 0.2, 0.75, 0.72)
        detector.update(1.1, 0.2, 0.2, 0.75, 0.72)
        events = detector.update(1.5, 0.62, 0.64, 0.24, 0.22)
        self.assertEqual([event.kind for event in events], ["blink"])

    def test_sustained_closed_blendshape_still_becomes_long_closure(self) -> None:
        detector = BlinkDetector(long_closure_seconds=0.8)
        detector.update(1.0, 0.2, 0.2, 0.75, 0.72)
        detector.update(1.1, 0.2, 0.2, 0.75, 0.72)
        events = detector.update(1.9, 0.18, 0.20, 0.78, 0.74)
        self.assertEqual([event.kind for event in events], ["long_eye_closure_started"])

    def test_rejects_one_frame_noise(self) -> None:
        detector = BlinkDetector(minimum_closed_seconds=0.05)
        detector.update(1.0, 0.3, 0.3)
        events = detector.update(1.03, 1.0, 1.0)
        self.assertEqual(events, [])

    def test_counts_one_frame_high_confidence_fast_blink(self) -> None:
        detector = BlinkDetector(minimum_closed_seconds=0.05)
        detector.update(1.0, 0.12, 0.14, 0.78, 0.74)
        events = detector.update(1.03, 1.0, 1.0, 0.1, 0.1)
        self.assertEqual([event.kind for event in events], ["blink"])

    def test_counts_shallow_single_frame_when_both_signals_are_strong(self) -> None:
        detector = BlinkDetector(minimum_closed_seconds=0.05)
        detector.update(1.0, 0.30, 0.32, 0.60, 0.58)
        events = detector.update(1.03, 1.0, 1.0, 0.1, 0.1)
        self.assertEqual([event.kind for event in events], ["blink"])

    def test_rejects_one_frame_deep_geometry_without_blinkshape_confirmation(self) -> None:
        detector = BlinkDetector(minimum_closed_seconds=0.05)
        detector.update(1.0, 0.12, 0.14, 0.35, 0.32)
        events = detector.update(1.03, 1.0, 1.0, 0.1, 0.1)
        self.assertEqual(events, [])

    def test_long_closure_is_not_counted_as_blink(self) -> None:
        detector = BlinkDetector(long_closure_seconds=0.8)
        detector.update(1.0, 0.3, 0.3)
        detector.update(1.1, 0.3, 0.3)
        started = detector.update(1.9, 0.3, 0.3)
        ended = detector.update(2.0, 1.0, 1.0)
        self.assertEqual([event.kind for event in started], ["long_eye_closure_started"])
        self.assertEqual([event.kind for event in ended], ["long_eye_closure_ended"])

    def test_missing_face_cancels_partial_closure(self) -> None:
        detector = BlinkDetector()
        detector.update(1.0, 0.3, 0.3)
        detector.update(1.1, None, None)
        self.assertEqual(detector.update(1.2, 1.0, 1.0), [])
        self.assertEqual(detector.state, "OPEN")


if __name__ == "__main__":
    unittest.main()
