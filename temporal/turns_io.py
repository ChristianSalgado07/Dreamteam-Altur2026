"""
Load and validate the JSON turn metadata (files/turns/turns/*.json).

Reuses preprocessing.audio_io for paths — only reads under files/, never
writes back into it.
"""
import json

from preprocessing.audio_io import AUDIO_DIR, JSON_DIR

# a turn's end may exceed the recording's own duration_s by a small amount
# before we flag it — durations were computed from n_samples/sample_rate
# and turn timestamps could be rounded slightly differently upstream.
TIMESTAMP_TOLERANCE_S = 0.05

# same-channel turns starting less than this many seconds before the
# previous one ends are flagged as an overlap, not a rounding artifact.
OVERLAP_TOLERANCE_S = 0.05


def _turn_problems(turn, duration_s):
    problems = []
    channel = turn.get("channel")
    start = turn.get("start")
    end = turn.get("end")

    if channel not in (0, 1):
        problems.append("unexpected_channel")
    if start is None or end is None:
        problems.append("missing_timestamp")
        return problems
    if start < 0 or end < 0:
        problems.append("negative_timestamp")
    if not (start < end):
        problems.append("start_not_before_end")
    if duration_s is not None and end > duration_s + TIMESTAMP_TOLERANCE_S:
        problems.append("exceeds_duration")
    return problems


def load_and_validate_turns(anon_id: str, duration_s: float = None):
    """
    Returns (turns, report).

    `turns` is a chronologically sorted list of {"channel","start","end"}
    dicts that passed validation. Turns that fail validation are DROPPED
    from this list but always recorded in report["issues"] — they are
    never silently discarded without a trace.

    report["status"] is one of "missing_file", "empty", "parse_error",
    or "valid". A "valid" file can still have entries in report["issues"]
    (e.g. one malformed turn among otherwise-good ones, or an overlap) —
    status only reflects whether the file itself could be read and parsed.
    """
    path = JSON_DIR / f"{anon_id}.json"
    if not path.exists():
        return [], {"status": "missing_file", "issues": []}

    if path.stat().st_size == 0:
        return [], {"status": "empty", "issues": []}

    try:
        data = json.loads(path.read_text())
        raw_turns = data.get("turns", [])
    except Exception as e:
        return [], {"status": "parse_error", "issues": [], "error": str(e)}

    issues = []
    valid_turns = []
    for t in raw_turns:
        problems = _turn_problems(t, duration_s)
        if problems:
            issues.append({"turn": t, "problems": problems})
        else:
            valid_turns.append({
                "channel": int(t["channel"]),
                "start": float(t["start"]),
                "end": float(t["end"]),
            })

    valid_turns.sort(key=lambda t: t["start"])

    for channel in (0, 1):
        ch_turns = sorted(
            (t for t in valid_turns if t["channel"] == channel),
            key=lambda t: t["start"],
        )
        for a, b in zip(ch_turns, ch_turns[1:]):
            if b["start"] < a["end"] - OVERLAP_TOLERANCE_S:
                issues.append({
                    "turn_pair": [a, b],
                    "problems": ["same_channel_overlap"],
                })

    return valid_turns, {"status": "valid", "issues": issues}


def validate_all(duration_by_id: dict) -> tuple:
    """
    Runs load_and_validate_turns over every recording in files/audio and
    returns (counts, per_file_reports) for the validation report.
    """
    audio_ids = sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))

    counts = {
        "total_json_files": len(audio_ids),
        "valid_turn_files": 0,
        "empty_json_files": 0,
        "invalid_json_files": 0,
        "files_with_invalid_timestamps": 0,
        "files_with_overlapping_turns": 0,
        "files_with_unexpected_channel": 0,
        "total_dropped_turns": 0,
    }
    per_file = {}

    for anon_id in audio_ids:
        turns, report = load_and_validate_turns(anon_id, duration_by_id.get(anon_id))
        per_file[anon_id] = report
        status = report["status"]

        if status == "empty":
            counts["empty_json_files"] += 1
            continue
        if status in ("missing_file", "parse_error"):
            counts["invalid_json_files"] += 1
            continue

        counts["valid_turn_files"] += 1
        dropped_turn_problems = [
            p for issue in report["issues"] if "turn" in issue for p in issue["problems"]
        ]
        pair_problems = [
            p for issue in report["issues"] if "turn_pair" in issue for p in issue["problems"]
        ]
        counts["total_dropped_turns"] += sum(1 for issue in report["issues"] if "turn" in issue)
        if any(p in ("negative_timestamp", "start_not_before_end", "exceeds_duration", "missing_timestamp")
               for p in dropped_turn_problems):
            counts["files_with_invalid_timestamps"] += 1
        if "same_channel_overlap" in pair_problems:
            counts["files_with_overlapping_turns"] += 1
        if "unexpected_channel" in dropped_turn_problems:
            counts["files_with_unexpected_channel"] += 1

    return counts, per_file
