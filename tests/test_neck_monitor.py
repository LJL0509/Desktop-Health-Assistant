import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from neck_monitor import (  # noqa: E402
    assess_posture,
    build_baseline,
    classify_relative_motion,
    classify_posture,
    data_issue_message,
    display_size_from_window_rect,
    format_reminder_countdown,
    IssueAccumulator,
    resolve_visibility_mode,
    LatestFrameWorker,
    PostureNotifier,
    POSTURE_NOTIFICATION_TEXT,
    point_in_rect,
    scaled_ui_rect,
    target_posture_state,
    WATER_BUTTON_RECT,
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

    def test_coordinated_forward_tolerates_small_ratio_boundary_noise(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": 0.165,
                "torso_growth": 0.076,
                "ratio_growth": 0.089,
                "face_y_change": 0.023,
                "torso_y_change": 0.0,
                "torso_area_growth": -0.079,
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

    def test_observed_farther_posture_is_not_rejected_for_small_torso_area(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": -0.276,
                "torso_growth": -0.276,
                "ratio_growth": 0.007,
                "face_y_change": 0.152,
                "torso_y_change": 0.046,
                "torso_area_growth": -0.604,
            }
        )
        self.assertEqual(motion, "WHOLE BODY BACK")

    def test_observed_farther_posture_tolerates_ratio_segmentation_noise(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": -0.307,
                "torso_growth": -0.350,
                "ratio_growth": 0.069,
                "face_y_change": 0.160,
                "torso_y_change": 0.060,
                "torso_area_growth": -0.505,
            }
        )
        self.assertEqual(motion, "WHOLE BODY BACK")

    def test_severe_contour_loss_with_relative_head_growth_stays_insufficient(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": -0.255,
                "torso_growth": -0.353,
                "ratio_growth": 0.146,
                "face_y_change": 0.156,
                "torso_y_change": 0.049,
                "torso_area_growth": -0.642,
            }
        )
        self.assertEqual(motion, "DATA INSUFFICIENT")

    def test_whole_body_motion_clears_forward_state(self) -> None:
        features = {"ratio_growth": 0.12}
        state = target_posture_state(
            "NECK FORWARD",
            "WHOLE BODY FORWARD",
            features,
        )
        self.assertEqual(state, "NECK NORMAL")

    def test_reliable_motion_result_is_independent_of_previous_state(self) -> None:
        features = {"ratio_growth": 0.12}
        for previous in ("NECK NORMAL", "NECK FORWARD"):
            with self.subTest(previous=previous):
                self.assertEqual(
                    target_posture_state(previous, "STABLE", features),
                    "NECK NORMAL",
                )
                self.assertEqual(
                    target_posture_state(previous, "HEAD FORWARD", features),
                    "NECK FORWARD",
                )

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

    def test_partial_view_keeps_strong_head_forward_evidence(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": 0.10,
                "torso_growth": -0.20,
                "ratio_growth": 0.39,
                "face_y_change": 0.04,
                "torso_y_change": 0.0,
                "torso_area_growth": -0.74,
            }
        )
        self.assertEqual(motion, "HEAD FORWARD")

    def test_partial_view_treats_contour_loss_as_non_forward_when_evidence_is_weak(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": -0.04,
                "torso_growth": -0.09,
                "ratio_growth": 0.05,
                "face_y_change": 0.01,
                "torso_y_change": 0.0,
                "torso_area_growth": -0.70,
            },
            partial_view=True,
        )
        self.assertEqual(motion, "STABLE")

    def test_valid_non_forward_movement_is_stable(self) -> None:
        motion = classify_relative_motion(
            {
                "face_growth": -0.05,
                "torso_growth": 0.04,
                "ratio_growth": -0.09,
                "face_y_change": 0.07,
                "torso_y_change": 0.0,
                "torso_area_growth": -0.12,
            }
        )
        self.assertEqual(motion, "STABLE")

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
    def test_default_thresholds_are_per_issue(self) -> None:
        neck = IssueAccumulator()
        neck.update(0.0, "neck_forward")
        self.assertEqual(neck.update(59.9, "neck_forward"), [])
        self.assertEqual(
            [event["event"] for event in neck.update(60.0, "neck_forward")],
            ["posture_alert"],
        )

        low = IssueAccumulator()
        low.update(0.0, "head_too_low")
        self.assertEqual(low.update(59.9, "head_too_low"), [])
        events = low.update(60.0, "head_too_low")
        self.assertEqual([event["event"] for event in events], ["posture_alert"])
        self.assertEqual(events[0]["threshold_seconds"], 60.0)

    def test_does_not_alert_before_continuous_threshold(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        events = tracker.update(9.9, "neck_forward")
        self.assertFalse(any(event["event"] == "posture_alert" for event in events))
        self.assertFalse(tracker.alerted)

    def test_does_not_repeat_before_three_minutes(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        first = tracker.update(10.0, "neck_forward")
        second = tracker.update(189.9, "neck_forward")
        self.assertEqual([event["event"] for event in first], ["posture_alert"])
        self.assertFalse(any(event["event"] == "posture_alert" for event in second))
        self.assertEqual(tracker.statistics["neck_forward"]["alert_count"], 1)

    def test_repeats_after_three_minutes_then_every_ten_minutes(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        tracker.update(10.0, "neck_forward")

        second = tracker.update(190.0, "neck_forward")
        before_third = tracker.update(789.9, "neck_forward")
        third = tracker.update(790.0, "neck_forward")

        self.assertTrue(second[0]["repeat"])
        self.assertEqual(before_third, [])
        self.assertTrue(third[0]["repeat"])
        self.assertEqual(tracker.statistics["neck_forward"]["alert_count"], 3)

    def test_recovery_resets_repeat_schedule(self) -> None:
        tracker = IssueAccumulator(alert_seconds=10.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        tracker.update(10.0, "neck_forward")
        tracker.update(13.0, None)

        tracker.update(20.0, "neck_forward")
        self.assertEqual(tracker.update(29.9, "neck_forward"), [])
        events = tracker.update(30.0, "neck_forward")

        self.assertEqual([event["event"] for event in events], ["posture_alert"])
        self.assertFalse(events[0]["repeat"])

    def test_direct_issue_switch_keeps_continuous_alert_timer(self) -> None:
        tracker = IssueAccumulator(alert_seconds=60.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        tracker.update(40.0, "neck_forward")
        switch_events = tracker.update(40.0, "head_too_low")

        self.assertEqual(
            [event["event"] for event in switch_events],
            ["posture_issue_ended", "posture_issue_started"],
        )
        alert_events = tracker.update(60.0, "head_too_low")
        self.assertEqual([event["event"] for event in alert_events], ["posture_alert"])
        self.assertEqual(alert_events[0]["issue"], "head_too_low")
        self.assertEqual(alert_events[0]["duration_seconds"], 60.0)

    def test_switch_at_threshold_alerts_for_new_issue(self) -> None:
        tracker = IssueAccumulator(alert_seconds=60.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        tracker.update(59.0, "neck_forward")

        events = tracker.update(60.0, "head_too_low")
        alert = next(event for event in events if event["event"] == "posture_alert")

        self.assertEqual(alert["issue"], "head_too_low")
        self.assertEqual(alert["duration_seconds"], 60.0)

    def test_switched_issues_keep_separate_report_durations(self) -> None:
        tracker = IssueAccumulator(alert_seconds=60.0, recovery_grace_seconds=2.0)
        tracker.update(0.0, "neck_forward")
        tracker.update(40.0, "head_too_low")
        tracker.update(70.0, "head_too_low")
        tracker.update(72.0, None)

        summary = tracker.summary()
        self.assertEqual(summary["neck_forward"]["total_seconds"], 40.0)
        self.assertEqual(summary["head_too_low"]["total_seconds"], 30.0)
        self.assertEqual(summary["neck_forward"]["episode_count"], 1)
        self.assertEqual(summary["head_too_low"]["episode_count"], 1)

    def test_repeat_alert_uses_issue_active_when_repeat_is_due(self) -> None:
        tracker = IssueAccumulator(
            alert_seconds=10.0,
            recovery_grace_seconds=2.0,
            first_repeat_seconds=20.0,
        )
        tracker.update(0.0, "neck_forward")
        tracker.update(10.0, "neck_forward")
        tracker.update(20.0, "head_too_low")

        events = tracker.update(30.0, "head_too_low")

        self.assertEqual([event["event"] for event in events], ["posture_alert"])
        self.assertEqual(events[0]["issue"], "head_too_low")
        self.assertTrue(events[0]["repeat"])

    def test_exposes_time_until_next_repeat_alert(self) -> None:
        tracker = IssueAccumulator(
            alert_seconds=10.0,
            first_repeat_seconds=180.0,
        )
        tracker.update(0.0, "neck_forward")
        tracker.update(10.0, "neck_forward")

        self.assertEqual(tracker.seconds_until_next_alert(10.0), 180.0)
        self.assertEqual(tracker.seconds_until_next_alert(70.5), 119.5)

    def test_formats_repeat_countdown_for_display(self) -> None:
        self.assertEqual(format_reminder_countdown(180.0), "3m 00s")
        self.assertEqual(format_reminder_countdown(179.1), "3m 00s")
        self.assertEqual(format_reminder_countdown(119.0), "1m 59s")

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


class PostureNotifierTest(unittest.TestCase):
    def test_has_distinct_behavior_guidance_for_each_issue(self) -> None:
        self.assertIn("头部前倾", POSTURE_NOTIFICATION_TEXT["neck_forward"][1])
        self.assertIn("抬高头部", POSTURE_NOTIFICATION_TEXT["head_too_low"][1])

    def test_uses_shared_topmost_popup(self) -> None:
        popup_notifier = Mock()
        notifier = PostureNotifier(popup_notifier)
        notifier.show("neck_forward")

        popup_notifier.show.assert_called_once_with(
            *POSTURE_NOTIFICATION_TEXT["neck_forward"],
        )


class VisibilityTransitionTest(unittest.TestCase):
    def test_maps_usable_gap_between_low_and_partial_to_partial(self) -> None:
        self.assertEqual(
            resolve_visibility_mode(
                "CONTOUR UNSTABLE",
                "TOO LOW",
                True,
                True,
                0.20,
            ),
            "PARTIAL",
        )
        self.assertEqual(
            resolve_visibility_mode(
                "CONTOUR UNSTABLE",
                "PARTIAL",
                True,
                True,
                0.20,
            ),
            "PARTIAL",
        )

    def test_data_loss_does_not_clear_too_low(self) -> None:
        self.assertEqual(
            resolve_visibility_mode(
                "CONTOUR UNSTABLE",
                "TOO LOW",
                False,
                True,
                0.0,
            ),
            "TOO LOW",
        )
        self.assertEqual(
            resolve_visibility_mode(
                "CONTOUR UNSTABLE",
                "TOO LOW",
                True,
                True,
                0.17,
            ),
            "TOO LOW",
        )

    def test_partial_can_transition_directly_to_too_low(self) -> None:
        self.assertEqual(
            resolve_visibility_mode(
                "CONTOUR UNSTABLE",
                "PARTIAL",
                True,
                True,
                0.14,
            ),
            "TOO LOW",
        )

    def test_gap_without_metrics_remains_insufficient(self) -> None:
        self.assertEqual(
            resolve_visibility_mode(
                "CONTOUR UNSTABLE",
                "PARTIAL",
                False,
                True,
                0.0,
            ),
            "CONTOUR UNSTABLE",
        )
        self.assertEqual(
            resolve_visibility_mode(
                "CONTOUR UNSTABLE",
                "FULL",
                True,
                False,
                0.0,
            ),
            "CONTOUR UNSTABLE",
        )


class MonitorControlsTest(unittest.TestCase):
    def test_water_button_hit_area_has_stable_bounds(self) -> None:
        self.assertTrue(point_in_rect(500, 425, WATER_BUTTON_RECT))
        self.assertFalse(point_in_rect(500, 90, WATER_BUTTON_RECT))

    def test_water_button_hit_area_scales_with_display(self) -> None:
        scaled = scaled_ui_rect(WATER_BUTTON_RECT, (2560, 1536))
        self.assertEqual(scaled, (1880, 1306, 2440, 1443))
        self.assertTrue(point_in_rect(2000, 1380, scaled))
        self.assertFalse(point_in_rect(2000, 280, scaled))

    def test_display_size_uses_valid_window_image_rect(self) -> None:
        self.assertEqual(
            display_size_from_window_rect((0, 30, 2560, 1536)),
            (2560, 1536),
        )
        self.assertEqual(
            display_size_from_window_rect((0, 0, 0, 0)),
            (640, 480),
        )


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
