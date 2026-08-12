import sys
import threading
import time
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from neck_monitor import (  # noqa: E402
    assess_posture,
    build_baseline,
    classify_relative_motion,
    classify_posture,
    data_issue_message,
    IssueAccumulator,
    LatestFrameWorker,
    target_posture_state,
)


BASE_METRICS = {
    "face_width": 0.25,
    "face_center_x": 0.48,
    "face_center_y": 0.52,
    "torso_width": 0.60,
    "torso_center_x": 0.47,
    "torso_center_y": 0.86,
    "torso_area": 0.16,
    "face_torso_ratio": 0.25 / 0.60,
    "head_clearance_ratio": 0.55,
    "chin_y": 0.75,
    "contour_coverage": 0.95,
    "profile_variation": 0.10,
    "contour_top_y": 0.76,
    "contour_bottom_y": 0.99,
    "anchor_y": 0.92,
    "anchor_left": 0.18,
    "anchor_right": 0.78,
}


def changed(**values: float) -> dict[str, float]:
    return {**BASE_METRICS, **values}


BASELINE = {
    "median": BASE_METRICS,
    "mad": {key: 0.001 for key in BASE_METRICS},
    "relative_mad": {
        "face_width": 0.004,
        "torso_width": 0.002,
        "face_torso_ratio": 0.004,
    },
}


class NeckMonitorClassificationTest(unittest.TestCase):
    def test_normal(self) -> None:
        self.assertEqual(classify_posture(BASELINE, changed())[0], "NECK NORMAL")

    def test_head_forward(self) -> None:
        current = changed(
            face_width=0.275,
            face_center_y=0.55,
            torso_width=0.58,
            torso_area=0.12,
            face_torso_ratio=0.275 / 0.58,
        )
        state, features, motion = assess_posture(BASELINE, current)
        self.assertEqual(state, "NECK FORWARD")
        self.assertEqual(motion, "HEAD FORWARD")
        self.assertGreater(features["ratio_growth"], 0.05)

    def test_whole_body_forward_is_normal(self) -> None:
        current = changed(
            face_width=0.30,
            torso_width=0.68,
            face_torso_ratio=0.30 / 0.68,
        )
        state, _, motion = assess_posture(BASELINE, current)
        self.assertEqual(state, "NECK NORMAL")
        self.assertEqual(motion, "WHOLE BODY FORWARD")

    def test_observed_coordinated_forward_motion_is_not_uncertain(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": 0.108,
                "torso_growth": 0.121,
                "ratio_growth": 0.005,
                "face_y_change": 0.052,
                "torso_y_change": 0.032,
                "torso_area_growth": -0.145,
            }
        )
        self.assertEqual(motion, "WHOLE BODY FORWARD")

    def test_observed_small_contour_jitter_is_stable(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": 0.032,
                "torso_growth": 0.074,
                "ratio_growth": -0.043,
                "face_y_change": 0.000,
                "torso_y_change": 0.003,
                "torso_area_growth": 0.050,
            }
        )
        self.assertEqual(motion, "STABLE")

    def test_observed_mild_forward_combined_signals_are_not_uncertain(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": 0.054,
                "torso_growth": -0.013,
                "ratio_growth": 0.060,
                "face_y_change": 0.028,
                "torso_y_change": 0.018,
                "torso_area_growth": -0.178,
            }
        )
        self.assertEqual(motion, "HEAD FORWARD")

    def test_head_forward_after_whole_body_moves_closer(self) -> None:
        current = changed(
            face_width=0.31,
            face_center_y=0.50,
            torso_width=0.66,
            torso_area=0.13,
            face_torso_ratio=0.31 / 0.66,
        )
        state, features, motion = assess_posture(BASELINE, current)
        self.assertEqual(state, "NECK FORWARD")
        self.assertEqual(motion, "HEAD FORWARD")
        self.assertGreater(features["ratio_growth"], 0.09)

    def test_head_forward_when_whole_body_is_farther(self) -> None:
        current = changed(
            face_width=0.225,
            face_center_y=0.60,
            torso_width=0.46,
            torso_center_y=0.89,
            torso_area=0.08,
            face_torso_ratio=0.225 / 0.46,
        )
        state, _, motion = assess_posture(BASELINE, current)
        self.assertEqual(state, "NECK FORWARD")
        self.assertEqual(motion, "HEAD FORWARD")

    def test_mild_head_forward_uses_combined_signals(self) -> None:
        current = changed(
            face_width=0.268,
            face_center_y=0.535,
            torso_width=0.60,
            torso_area=0.14,
            face_torso_ratio=0.268 / 0.60,
        )
        state, _, motion = assess_posture(BASELINE, current)
        self.assertEqual(state, "NECK FORWARD")
        self.assertEqual(motion, "HEAD FORWARD")

    def test_whole_body_back_is_normal(self) -> None:
        current = changed(
            face_width=0.225,
            torso_width=0.54,
            face_torso_ratio=0.225 / 0.54,
        )
        state, _, motion = assess_posture(BASELINE, current)
        self.assertEqual(state, "NECK NORMAL")
        self.assertEqual(motion, "WHOLE BODY BACK")

    def test_forward_state_stays_latched_during_whole_body_motion(self) -> None:
        features = {"ratio_growth": 0.12}
        state = target_posture_state(
            "NECK FORWARD",
            "WHOLE BODY FORWARD",
            features,
        )
        self.assertEqual(state, "NECK FORWARD")

    def test_forward_state_recovers_near_reference_ratio(self) -> None:
        features = {"ratio_growth": 0.03}
        state = target_posture_state("NECK FORWARD", "STABLE", features)
        self.assertEqual(state, "NECK NORMAL")

    def test_uncertain_data_does_not_clear_forward_state(self) -> None:
        features = {"ratio_growth": -0.10}
        state = target_posture_state("NECK FORWARD", "UNCERTAIN", features)
        self.assertEqual(state, "NECK FORWARD")

    def test_severe_contour_loss_is_data_insufficient(self) -> None:
        current = changed(
            face_width=0.28,
            torso_width=0.38,
            torso_area=0.05,
            face_torso_ratio=0.28 / 0.38,
        )
        state, _, motion = assess_posture(BASELINE, current)
        self.assertEqual(state, "NECK NORMAL")
        self.assertEqual(motion, "DATA INSUFFICIENT")

    def test_builds_stable_contour_baseline(self) -> None:
        samples = [changed(face_width=0.25 + (index % 3 - 1) * 0.0005) for index in range(40)]
        baseline, message = build_baseline(samples)
        self.assertIsNotNone(baseline)
        self.assertEqual(message, "ok")

    def test_rejects_too_few_calibration_samples(self) -> None:
        baseline, message = build_baseline([changed()] * 20)
        self.assertIsNone(baseline)
        self.assertEqual(message, "not enough valid contour data")

    def test_data_issue_message_for_low_posture(self) -> None:
        self.assertEqual(
            data_issue_message("head_too_low"),
            "HEAD TOO LOW - RAISE IT SLIGHTLY",
        )

    def test_data_issue_message_for_missing_face(self) -> None:
        self.assertEqual(
            data_issue_message("face_missing"),
            "FACE NOT VISIBLE - MOVE BACK OR SIT UP",
        )


class IssueAccumulatorTest(unittest.TestCase):
    def test_does_not_alert_before_continuous_threshold(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        events = tracker.update(9.9, "neck_forward")
        self.assertFalse(any(event["event"] == "posture_alert" for event in events))
        self.assertFalse(tracker.alerted)

    def test_alerts_once_after_continuous_threshold(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        first = tracker.update(10.0, "neck_forward")
        second = tracker.update(20.0, "neck_forward")
        self.assertEqual([event["event"] for event in first], ["posture_alert"])
        self.assertFalse(any(event["event"] == "posture_alert" for event in second))
        self.assertEqual(tracker.statistics["neck_forward"]["alert_count"], 1)

    def test_short_normal_jitter_does_not_split_episode(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        tracker.update(4.0, "neck_forward")
        self.assertEqual(tracker.update(5.0, None), [])
        self.assertEqual(tracker.update(6.0, "neck_forward"), [])
        self.assertEqual(tracker.current_issue, "neck_forward")

    def test_completed_episode_updates_report_statistics(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "head_too_low")
        tracker.update(12.0, "head_too_low")
        events = tracker.update(15.0, None)
        ended = next(event for event in events if event["event"] == "posture_issue_ended")
        self.assertEqual(ended["duration_seconds"], 12.0)
        summary = tracker.summary()["head_too_low"]
        self.assertEqual(summary["episode_count"], 1)
        self.assertEqual(summary["total_seconds"], 12.0)
        self.assertEqual(summary["longest_seconds"], 12.0)
        self.assertEqual(summary["alert_count"], 1)


class LatestFrameWorkerTest(unittest.TestCase):
    def test_processes_payload_and_publishes_sequence(self) -> None:
        worker = LatestFrameWorker(lambda: lambda value: value * 2)
        try:
            sequence = worker.submit(4)
            deadline = time.monotonic() + 1.0
            latest_sequence, value = worker.latest()
            while latest_sequence != sequence and time.monotonic() < deadline:
                time.sleep(0.01)
                latest_sequence, value = worker.latest()
            self.assertEqual(latest_sequence, sequence)
            self.assertEqual(value, 8)
        finally:
            worker.close()

    def test_pending_payload_is_replaced_by_newest_frame(self) -> None:
        started = threading.Event()
        release = threading.Event()
        processed = []

        def make_processor():
            def process(value):
                processed.append(value)
                if value == 1:
                    started.set()
                    release.wait(timeout=1.0)
                return value
            return process

        worker = LatestFrameWorker(make_processor)
        try:
            worker.submit(1)
            self.assertTrue(started.wait(timeout=1.0))
            worker.submit(2)
            newest_sequence = worker.submit(3)
            release.set()
            deadline = time.monotonic() + 1.0
            latest_sequence, value = worker.latest()
            while latest_sequence != newest_sequence and time.monotonic() < deadline:
                time.sleep(0.01)
                latest_sequence, value = worker.latest()
            self.assertEqual(processed, [1, 3])
            self.assertEqual(value, 3)
        finally:
            release.set()
            worker.close()

    def test_close_stops_background_thread(self) -> None:
        worker = LatestFrameWorker(lambda: lambda value: value)
        worker.close()
        self.assertFalse(worker.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
