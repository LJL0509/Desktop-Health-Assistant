import argparse
import json
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

from app_paths import data_path


MONITOR_DIR = data_path("monitor-sessions")
REMINDER_DIR = data_path("reminders")
ACTIVITY_DIR = data_path("activity")
REPORT_DIR = data_path("reports")
HYDRATION_INTERVAL_SECONDS = 60 * 60
ISSUE_LABELS = {
    "neck_forward": "头部前伸",
    "head_too_low": "头部过低",
}
ISSUE_GUIDANCE = {
    "neck_forward": "优先调整屏幕高度和坐姿，让头部回到躯干上方；短暂活动后重新坐稳。",
    "head_too_low": "减少持续低头，适当抬高屏幕或正在阅读的内容。",
}


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    events = []
    invalid_lines = 0
    if not path.exists():
        return events, invalid_lines
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            payload["_source"] = str(path)
            events.append(payload)
        except (json.JSONDecodeError, TypeError):
            invalid_lines += 1
    return events, invalid_lines


def event_time(event: dict) -> datetime:
    return datetime.fromisoformat(event["timestamp"])


def day_bounds(target: date, timezone) -> tuple[datetime, datetime]:
    start = datetime.combine(target, time.min, tzinfo=timezone)
    return start, start + timedelta(days=1)


def overlap_seconds(
    start: datetime,
    end: datetime,
    day_start: datetime,
    day_end: datetime,
) -> float:
    return max(0.0, (min(end, day_end) - max(start, day_start)).total_seconds())


def pause_seconds(events: list[dict], start: datetime, end: datetime) -> float:
    paused_at = None
    total = 0.0
    for event in sorted(events, key=event_time):
        if event["event"] == "camera_paused" and paused_at is None:
            paused_at = event_time(event)
        elif event["event"] == "camera_resumed" and paused_at is not None:
            total += max(0.0, (event_time(event) - paused_at).total_seconds())
            paused_at = None
    if paused_at is not None:
        total += max(0.0, (end - paused_at).total_seconds())
    return min(total, max(0.0, (end - start).total_seconds()))


def session_interval(events: list[dict]) -> tuple[datetime, datetime, bool, bool]:
    ordered = sorted(events, key=event_time)
    first = event_time(ordered[0])
    last = event_time(ordered[-1])
    starts = [event_time(event) for event in ordered if event["event"] == "session_started"]
    ends = [event_time(event) for event in ordered if event["event"] == "session_ended"]
    summaries = [event for event in ordered if event["event"] == "session_summary"]

    start = starts[0] if starts else first
    end = ends[-1] if ends else last
    used_estimate = not starts or not ends
    if summaries and not starts:
        summary_end = event_time(summaries[-1])
        start = summary_end - timedelta(
            seconds=max(0.0, float(summaries[-1].get("session_seconds", 0.0)))
        )
    return start, end, bool(ends), used_estimate


def summarize_monitoring(
    target: date,
    monitor_dir: Path = MONITOR_DIR,
) -> tuple[dict, dict[str, dict], dict, list[str]]:
    sessions = []
    invalid_lines = 0
    timezone = datetime.now().astimezone().tzinfo
    day_start, day_end = day_bounds(target, timezone)
    issue_stats = {
        issue: {
            "total_seconds": 0.0,
            "episode_count": 0,
            "longest_seconds": 0.0,
            "alert_count": 0,
        }
        for issue in ISSUE_LABELS
    }
    camera_missing_seconds = 0.0
    posture_unreliable_seconds = 0.0
    open_issue_count = 0
    blink_statistics = {
        "blink_count": 0,
        "valid_observation_seconds": 0.0,
        "average_rate_per_minute": None,
        "low_rate_alert_count": 0,
        "long_closure_count": 0,
        "long_closure_seconds": 0.0,
        "session_count": 0,
    }

    for path in sorted(monitor_dir.glob("*.jsonl")) if monitor_dir.exists() else []:
        events, errors = read_jsonl(path)
        invalid_lines += errors
        if not events:
            continue
        try:
            start, end, complete, estimated = session_interval(events)
        except (KeyError, ValueError):
            invalid_lines += 1
            continue
        duration = overlap_seconds(start, end, day_start, day_end)
        if duration <= 0:
            continue
        relevant = [
            event
            for event in events
            if day_start <= event_time(event) < day_end
        ]
        paused = pause_seconds(relevant, max(start, day_start), min(end, day_end))
        sessions.append(
            {
                "duration_seconds": duration,
                "paused_seconds": min(paused, duration),
                "complete": complete,
                "estimated": estimated,
            }
        )

        active_issues = defaultdict(int)
        for event in relevant:
            name = event.get("event")
            issue = event.get("issue")
            if name == "posture_issue_started" and issue in issue_stats:
                active_issues[issue] += 1
            elif name == "posture_issue_ended" and issue in issue_stats:
                duration_seconds = max(0.0, float(event.get("duration_seconds", 0.0)))
                stats = issue_stats[issue]
                stats["total_seconds"] += duration_seconds
                stats["episode_count"] += 1
                stats["longest_seconds"] = max(stats["longest_seconds"], duration_seconds)
                active_issues[issue] = max(0, active_issues[issue] - 1)
            elif name == "posture_alert" and issue in issue_stats:
                issue_stats[issue]["alert_count"] += 1
            elif name == "data_recovered":
                camera_missing_seconds += max(0.0, float(event.get("missing_seconds", 0.0)))
            elif name == "posture_data_recovered":
                posture_unreliable_seconds += max(
                    0.0, float(event.get("duration_seconds", 0.0))
                )
            elif name == "session_summary" and event.get("blink_statistics"):
                blink = event["blink_statistics"]
                blink_statistics["blink_count"] += max(
                    0, int(blink.get("blink_count", 0))
                )
                blink_statistics["valid_observation_seconds"] += max(
                    0.0, float(blink.get("valid_observation_seconds", 0.0))
                )
                blink_statistics["low_rate_alert_count"] += max(
                    0, int(blink.get("low_rate_alert_count", 0))
                )
                blink_statistics["long_closure_count"] += max(
                    0, int(blink.get("long_closure_count", 0))
                )
                blink_statistics["long_closure_seconds"] += max(
                    0.0, float(blink.get("long_closure_seconds", 0.0))
                )
                blink_statistics["session_count"] += 1
        open_issue_count += sum(active_issues.values())

    total_seconds = sum(session["duration_seconds"] for session in sessions)
    paused_total = sum(session["paused_seconds"] for session in sessions)
    active_seconds = max(0.0, total_seconds - paused_total)
    camera_missing_seconds = min(camera_missing_seconds, active_seconds)
    valid_seconds = max(0.0, active_seconds - camera_missing_seconds)
    coverage_ratio = valid_seconds / active_seconds if active_seconds else None
    notes = []
    incomplete_count = sum(not session["complete"] for session in sessions)
    estimated_count = sum(session["estimated"] for session in sessions)
    if incomplete_count:
        notes.append(f"有 {incomplete_count} 个监控会话没有正常结束，部分时长可能低估。")
    if estimated_count:
        notes.append(f"有 {estimated_count} 个会话缺少完整起止事件，运行时长使用日志估算。")
    if open_issue_count:
        notes.append(f"有 {open_issue_count} 段姿势问题没有结束事件，未计入问题时长。")
    if invalid_lines:
        notes.append(f"忽略了 {invalid_lines} 行无法解析的监控日志。")

    monitoring = {
        "session_count": len(sessions),
        "complete_session_count": len(sessions) - incomplete_count,
        "run_seconds": total_seconds,
        "camera_paused_seconds": paused_total,
        "active_camera_seconds": active_seconds,
        "camera_data_missing_seconds": camera_missing_seconds,
        "posture_unreliable_seconds": posture_unreliable_seconds,
        "valid_monitoring_seconds": valid_seconds,
        "data_coverage_ratio": coverage_ratio,
    }
    blink_seconds = blink_statistics["valid_observation_seconds"]
    if blink_seconds > 0:
        blink_statistics["average_rate_per_minute"] = (
            blink_statistics["blink_count"] * 60.0 / blink_seconds
        )
    return monitoring, issue_stats, blink_statistics, notes


def summarize_reminders(
    target: date,
    reminder_dir: Path = REMINDER_DIR,
) -> tuple[dict, list[str]]:
    path = reminder_dir / f"{target.isoformat()}.jsonl"
    events, invalid_lines = read_jsonl(path)
    events = sorted(
        (event for event in events if not event.get("event", "").startswith("test_")),
        key=event_time,
    )
    water_events = [event for event in events if event.get("event") == "water_recorded"]
    hydration_events = [
        event for event in events if event.get("event") == "hydration_reminder"
    ]
    sleep_events = [event for event in events if event.get("event") == "sleep_reminder"]
    sleep_acknowledged = [
        event
        for event in events
        if event.get("event") == "reminder_action"
        and event.get("action") == "sleep_acknowledged"
    ]

    longest_reminder_streak = 0
    current_streak = 0
    for event in events:
        if event.get("event") == "hydration_reminder":
            current_streak += 1
            longest_reminder_streak = max(longest_reminder_streak, current_streak)
        elif event.get("event") == "water_recorded":
            current_streak = 0

    notes = []
    if invalid_lines:
        notes.append(f"忽略了 {invalid_lines} 行无法解析的提醒日志。")
    if hydration_events:
        notes.append("未记录喝水时长按连续补水提醒估算，不代表摄像头识别了喝水行为。")
    return {
        "water_record_count": len(water_events),
        "water_total_ml": sum(max(0, int(event.get("amount_ml", 0))) for event in water_events),
        "hydration_reminder_count": len(hydration_events),
        "longest_unconfirmed_water_seconds": (
            longest_reminder_streak * HYDRATION_INTERVAL_SECONDS
        ),
        "hydration_confirmation_pending": current_streak > 0,
        "sleep_reminder_count": len(sleep_events),
        "sleep_acknowledged_count": len(sleep_acknowledged),
    }, notes


def summarize_activity(
    target: date,
    activity_dir: Path = ACTIVITY_DIR,
    now: datetime | None = None,
) -> tuple[dict, list[str]]:
    timezone = datetime.now().astimezone().tzinfo
    day_start, day_end = day_bounds(target, timezone)
    generated_at = now or datetime.now().astimezone()
    cutoff = min(generated_at, day_end) if target == generated_at.date() else day_end
    paths = [
        activity_dir / f"{target.isoformat()}.jsonl",
        activity_dir / f"{(target + timedelta(days=1)).isoformat()}.jsonl",
    ]
    events = []
    invalid_lines = 0
    for path in paths:
        file_events, errors = read_jsonl(path)
        events.extend(file_events)
        invalid_lines += errors
    events.sort(key=event_time)

    completed_starts = set()
    durations = []
    for event in events:
        if event.get("event") != "continuous_use_ended":
            continue
        ended_at = event_time(event)
        started_value = event.get("started_at")
        if started_value:
            started_at = datetime.fromisoformat(started_value)
            completed_starts.add(started_value)
            duration = overlap_seconds(started_at, ended_at, day_start, day_end)
        elif day_start <= ended_at < day_end:
            duration = max(0.0, float(event.get("duration_seconds", 0.0)))
        else:
            continue
        if duration > 0:
            durations.append(duration)

    open_starts = []
    for event in events:
        if event.get("event") != "continuous_use_started":
            continue
        started_at = event_time(event)
        started_value = started_at.isoformat(timespec="seconds")
        if started_value in completed_starts:
            continue
        if started_at < cutoff and started_at < day_end:
            duration = overlap_seconds(started_at, cutoff, day_start, day_end)
            if duration > 0:
                durations.append(duration)
                open_starts.append(started_at)

    notes = [
        "连续使用由键盘鼠标空闲时间估算；看视频或阅读时，长时间无输入可能被误认为已经离开。",
        "连续 5 分钟无输入后重置使用时长；该值是可调整的产品默认值。",
    ]
    if open_starts:
        notes.append(
            f"有 {len(open_starts)} 段连续使用尚未记录结束，时长统计到报告生成时刻。"
        )
    if invalid_lines:
        notes.append(f"忽略了 {invalid_lines} 行无法解析的活动日志。")
    return {
        "session_count": len(durations),
        "total_seconds": sum(durations),
        "longest_seconds": max(durations, default=0.0),
        "open_session_count": len(open_starts),
    }, notes


def build_guidance(report: dict) -> list[str]:
    guidance = []
    monitoring = report["monitoring"]
    activity = report["computer_activity"]
    has_observation = (
        monitoring["session_count"] > 0
        or activity["session_count"] > 0
        or activity["open_session_count"] > 0
        or report["blink_statistics"]["valid_observation_seconds"] > 0
    )
    issues = report["posture_issues"]
    ranked_issues = sorted(
        issues,
        key=lambda issue: issues[issue]["total_seconds"],
        reverse=True,
    )
    for issue in ranked_issues:
        if issues[issue]["total_seconds"] > 0:
            guidance.append(ISSUE_GUIDANCE[issue])
    reminders = report["reminders"]
    if reminders["hydration_reminder_count"]:
        guidance.append("补水提醒出现后及时喝水并点击“已喝水”，这样报告才能结束未确认时段。")
    if activity["longest_seconds"] >= 60 * 60:
        guidance.append("连续使用电脑时间较长，安排一次至少 5 分钟的离屏活动，再开始下一段任务。")
    blink = report["blink_statistics"]
    if blink["low_rate_alert_count"]:
        guidance.append("今天出现过持续低眨眼频率提示，工作时定期看向远处，并有意识地自然眨眼。")
    coverage = report["monitoring"]["data_coverage_ratio"]
    if coverage is not None and coverage < 0.8:
        guidance.append("今天有效数据覆盖不足 80%，先检查坐姿是否过低、脸部是否离开画面或摄像头是否被占用。")
    if not has_observation:
        guidance.insert(0, "当天没有有效监测数据，无法评价电脑使用、姿势或眨眼情况。")
    elif not guidance:
        guidance.append("已监测时段内没有记录到持续姿势问题或未处理的补水提醒。")
    return guidance


def build_report(
    target: date,
    monitor_dir: Path = MONITOR_DIR,
    reminder_dir: Path = REMINDER_DIR,
    activity_dir: Path = ACTIVITY_DIR,
    generated_at: datetime | None = None,
) -> dict:
    generated_at = generated_at or datetime.now().astimezone()
    monitoring, posture_issues, blink_statistics, monitor_notes = summarize_monitoring(
        target, monitor_dir
    )
    reminders, reminder_notes = summarize_reminders(target, reminder_dir)
    computer_activity, activity_notes = summarize_activity(
        target, activity_dir, generated_at
    )
    report = {
        "date": target.isoformat(),
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "monitoring": monitoring,
        "posture_issues": posture_issues,
        "blink_statistics": blink_statistics,
        "reminders": reminders,
        "computer_activity": computer_activity,
        "guidance": [],
        "data_notes": monitor_notes + reminder_notes + activity_notes,
        "not_yet_monitored": [
            "真实坐姿与久坐状态",
            "固定姿势时长",
            "屏幕距离",
        ],
        "medical_notice": "本报告是健康行为提示，不是医学诊断。",
    }
    report["guidance"] = build_guidance(report)
    return report


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "暂无数据"
    total = max(0, round(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分钟"
    if minutes:
        return f"{minutes} 分钟 {secs} 秒"
    return f"{secs} 秒"


def render_markdown(report: dict) -> str:
    monitoring = report["monitoring"]
    reminders = report["reminders"]
    activity = report["computer_activity"]
    blink = report["blink_statistics"]
    blink_rate_text = (
        "暂无数据"
        if blink["average_rate_per_minute"] is None
        else f"{blink['average_rate_per_minute']:.1f} 次/分钟"
    )
    coverage = monitoring["data_coverage_ratio"]
    coverage_text = f"{coverage:.1%}" if coverage is not None else "暂无数据"
    lines = [
        f"# {report['date']} 每日健康提示报告",
        "",
        f"> {report['medical_notice']}",
        "",
        "## 今日重点",
        "",
        *[f"- {item}" for item in report["guidance"]],
        "",
        "## 监控覆盖",
        "",
        f"- 监控会话：{monitoring['session_count']} 次，其中完整结束 {monitoring['complete_session_count']} 次",
        f"- 程序运行时长：{format_duration(monitoring['run_seconds'])}",
        f"- 有效监测时长：{format_duration(monitoring['valid_monitoring_seconds'])}",
        f"- 摄像头数据覆盖率：{coverage_text}",
        f"- 摄像头暂停：{format_duration(monitoring['camera_paused_seconds'])}",
        f"- 可确定的数据缺失：{format_duration(monitoring['camera_data_missing_seconds'])}",
        "",
        "## 连续电脑使用",
        "",
        f"- 记录到的使用段数：{activity['session_count']} 次",
        f"- 累计使用时长：{format_duration(activity['total_seconds'])}",
        f"- 最长连续使用：{format_duration(activity['longest_seconds'])}",
        "",
        "## 眨眼与眼部观察",
        "",
        f"- 有效眼部观察时长：{format_duration(blink['valid_observation_seconds'])}",
        f"- 记录到的眨眼：{blink['blink_count']} 次",
        f"- 平均眨眼频率：{blink_rate_text}",
        f"- 低频提醒：{blink['low_rate_alert_count']} 次",
        f"- 持续闭眼：{blink['long_closure_count']} 次，共 {format_duration(blink['long_closure_seconds'])}",
        "",
        "## 姿势问题",
        "",
        "| 类型 | 累计时长 | 次数 | 最长一次 | 提醒次数 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for issue, label in ISSUE_LABELS.items():
        stats = report["posture_issues"][issue]
        lines.append(
            f"| {label} | {format_duration(stats['total_seconds'])} | "
            f"{stats['episode_count']} | {format_duration(stats['longest_seconds'])} | "
            f"{stats['alert_count']} |"
        )
    lines.extend(
        [
            "",
            "## 补水与作息提醒",
            "",
            f"- 已记录喝水：{reminders['water_record_count']} 次，约 {reminders['water_total_ml']} ml",
            f"- 补水提醒：{reminders['hydration_reminder_count']} 次",
            f"- 最长未确认喝水：{format_duration(reminders['longest_unconfirmed_water_seconds'])}",
            f"- 当前是否仍有未确认补水提醒：{'是' if reminders['hydration_confirmation_pending'] else '否'}",
            f"- 凌晨睡眠提醒：{reminders['sleep_reminder_count']} 次，确认 {reminders['sleep_acknowledged_count']} 次",
            "",
            "## 数据边界",
            "",
            "以下项目尚未接入，因此本报告不判断其是否健康：",
            "",
            *[f"- {item}" for item in report["not_yet_monitored"]],
        ]
    )
    if report["data_notes"]:
        lines.extend(
            [
                "",
                "数据质量说明：",
                "",
                *[f"- {item}" for item in report["data_notes"]],
            ]
        )
    lines.extend(["", f"生成时间：{report['generated_at']}", ""])
    return "\n".join(lines)


def write_report(report: dict, output_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = report["date"]
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


class AutoReportScheduler:
    def __init__(
        self,
        started_at: datetime,
        output_dir: Path = REPORT_DIR,
        generator: Callable[[date, datetime], dict] | None = None,
    ) -> None:
        if started_at.tzinfo is None:
            raise ValueError("started_at must be timezone-aware")
        self.current_date = started_at.date()
        self.output_dir = output_dir
        self.generator = generator or (
            lambda target, generated_at: build_report(
                target,
                generated_at=generated_at,
            )
        )

    def report_exists(self, target: date) -> bool:
        stem = target.isoformat()
        return (
            (self.output_dir / f"{stem}.json").exists()
            and (self.output_dir / f"{stem}.md").exists()
        )

    def generate_if_missing(self, target: date, now: datetime) -> bool:
        self._validate_now(now)
        if self.report_exists(target):
            return False
        report = self.generator(target, now)
        write_report(report, self.output_dir)
        return True

    def generate_previous_day_if_missing(self, now: datetime) -> list[date]:
        self._validate_now(now)
        target = now.date() - timedelta(days=1)
        return [target] if self.generate_if_missing(target, now) else []

    def poll(
        self,
        now: datetime,
        close_current_day: Callable[[date, datetime], None] | None = None,
    ) -> list[date]:
        self._validate_now(now)
        if now.date() <= self.current_date:
            return []

        if close_current_day is not None:
            rollover_at = datetime.combine(
                self.current_date + timedelta(days=1),
                time.min,
                tzinfo=now.tzinfo,
            )
            close_current_day(self.current_date, rollover_at)

        generated = []
        target = self.current_date
        while target < now.date():
            if self.generate_if_missing(target, now):
                generated.append(target)
            target += timedelta(days=1)
        self.current_date = now.date()
        return generated

    @staticmethod
    def _validate_now(now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a daily health report")
    parser.add_argument("--date", type=parse_date, default=date.today())
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(args.date)
    json_path, markdown_path = write_report(report, args.output_dir)
    print(f"JSON report: {json_path}")
    print(f"Readable report: {markdown_path}")


if __name__ == "__main__":
    main()
