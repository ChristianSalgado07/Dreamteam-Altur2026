"""
Turn-level, turn-taking, response-latency, within-turn-pause, and
temporal-variability feature computations for one recording's validated
turn list. See temporal/README.md for the methodology behind each piece.
"""
import numpy as np

from preprocessing.audio_io import EXPECTED_SAMPLE_RATE
from preprocessing.features import HOP_LENGTH

# Frame hop of the existing acoustic time-series (outputs/features/timeseries.csv),
# reused here so within-turn-pause detection lines up with those frames exactly.
FRAME_HOP_S = HOP_LENGTH / EXPECTED_SAMPLE_RATE  # 0.032s

# "nearly zero" response-latency / gap tolerance — a target start within this
# many seconds of the other speaker's turn end counts as neither a clear
# pause nor an overlap, just an immediate (floor-taking) transition.
NEAR_ZERO_GAP_TOLERANCE_S = 0.05

# fixed, documented thresholds for "very short" / "long" target turns
# (Step 8) — deliberately simple fixed cutoffs rather than data-driven
# ones, per rules.md's "do not overengineer".
SHORT_TURN_THRESHOLD_S = 0.5
LONG_TURN_THRESHOLD_S = 10.0

# within-turn candidate-pause detection (Step 7): a frame counts as
# "low energy" if its RMS is below this percentile of the RMS values
# seen across ALL of this recording's target turns (i.e. relative to
# how loud this speaker is when they are actually the one talking —
# not relative to the whole recording, which is mostly silence from
# Channel 0's perspective whenever the other speaker has the floor).
PAUSE_RMS_PERCENTILE = 20
MIN_PAUSE_DURATION_S = 0.15


def _duration_stats(turns: list, prefix: str) -> dict:
    durations = np.array([t["end"] - t["start"] for t in turns], dtype=float)
    if len(durations) == 0:
        return {
            f"{prefix}_turn_count": 0,
            f"{prefix}_total_speaking_time": 0.0,
            f"{prefix}_mean_turn_duration": np.nan,
            f"{prefix}_median_turn_duration": np.nan,
            f"{prefix}_std_turn_duration": np.nan,
            f"{prefix}_min_turn_duration": np.nan,
            f"{prefix}_max_turn_duration": np.nan,
        }
    return {
        f"{prefix}_turn_count": int(len(durations)),
        f"{prefix}_total_speaking_time": float(np.sum(durations)),
        f"{prefix}_mean_turn_duration": float(np.mean(durations)),
        f"{prefix}_median_turn_duration": float(np.median(durations)),
        f"{prefix}_std_turn_duration": float(np.std(durations)),
        f"{prefix}_min_turn_duration": float(np.min(durations)),
        f"{prefix}_max_turn_duration": float(np.max(durations)),
    }


def target_turn_stats(target_turns: list) -> dict:
    return _duration_stats(target_turns, "target")


def other_turn_stats(other_turns: list) -> dict:
    return _duration_stats(other_turns, "other")


def speaking_ratio_stats(target_turns: list, all_turns: list, duration_s: float) -> dict:
    """
    target_speaking_ratio uses the conversation's own time span (first
    turn start -> last turn end across BOTH channels) as the denominator,
    not the raw wav duration_s — a recording with a long silent lead-in
    or trailing silence would otherwise understate how much of the
    *actual conversation* the target speaker filled.

    target_speaking_ratio_full_duration (denominator = wav duration_s) is
    kept alongside it only as a reference value / for confound checks.
    """
    target_total = sum(t["end"] - t["start"] for t in target_turns)
    if all_turns:
        span = max(t["end"] for t in all_turns) - min(t["start"] for t in all_turns)
    else:
        span = np.nan
    ratio_span = target_total / span if span and span > 0 else np.nan
    ratio_full = target_total / duration_s if duration_s else np.nan
    return {
        "conversation_span_s": span,
        "target_speaking_ratio": ratio_span,
        "target_speaking_ratio_full_duration": ratio_full,
    }


def build_merged_sequence(all_turns: list) -> list:
    return sorted(all_turns, key=lambda t: t["start"])


def transition_stats(seq: list) -> dict:
    """
    Walks the chronologically-merged (both channels) turn sequence and
    classifies each consecutive pair. `inter_turn_gap = next.start -
    previous.end` — negative values (overlapping speech) are kept as-is,
    not clipped to zero.
    """
    n = len(seq)
    if n < 2:
        return {
            "total_transitions": 0,
            "target_to_other_transitions": 0,
            "other_to_target_transitions": 0,
            "target_to_target_transitions": 0,
            "other_to_other_transitions": 0,
            "proportion_target_followed_by_other": np.nan,
            "proportion_other_followed_by_target": np.nan,
            "inter_turn_gap_mean": np.nan,
            "inter_turn_gap_median": np.nan,
            "inter_turn_gap_std": np.nan,
            "inter_turn_gap_min": np.nan,
            "inter_turn_gap_max": np.nan,
        }, []

    gaps = []
    t2o = o2t = t2t = o2o = 0
    for a, b in zip(seq, seq[1:]):
        gaps.append(b["start"] - a["end"])
        if a["channel"] == 0 and b["channel"] == 1:
            t2o += 1
        elif a["channel"] == 1 and b["channel"] == 0:
            o2t += 1
        elif a["channel"] == 0 and b["channel"] == 0:
            t2t += 1
        else:
            o2o += 1

    gaps_arr = np.array(gaps, dtype=float)
    target_with_next = t2o + t2t
    other_with_next = o2t + o2o
    stats = {
        "total_transitions": n - 1,
        "target_to_other_transitions": t2o,
        "other_to_target_transitions": o2t,
        "target_to_target_transitions": t2t,
        "other_to_other_transitions": o2o,
        "proportion_target_followed_by_other": (t2o / target_with_next) if target_with_next else np.nan,
        "proportion_other_followed_by_target": (o2t / other_with_next) if other_with_next else np.nan,
        "inter_turn_gap_mean": float(np.mean(gaps_arr)),
        "inter_turn_gap_median": float(np.median(gaps_arr)),
        "inter_turn_gap_std": float(np.std(gaps_arr)),
        "inter_turn_gap_min": float(np.min(gaps_arr)),
        "inter_turn_gap_max": float(np.max(gaps_arr)),
    }
    return stats, gaps


def response_latency_stats(seq: list) -> tuple:
    """
    Response latency = target_start - other_end, computed only for
    other -> target transitions in the merged sequence (i.e. the target
    speaker taking the floor right after the other speaker stops).
    Negative values (the target started before the other speaker
    finished — overlapping/interrupting speech) are preserved, not
    clipped to zero.
    """
    latencies = []
    for a, b in zip(seq, seq[1:]):
        if a["channel"] == 1 and b["channel"] == 0:
            latencies.append(b["start"] - a["end"])

    if not latencies:
        return {
            "response_latency_mean": np.nan, "response_latency_median": np.nan,
            "response_latency_std": np.nan, "response_latency_min": np.nan,
            "response_latency_max": np.nan, "response_latency_n": 0,
            "response_latency_n_positive": 0, "response_latency_n_near_zero": 0,
            "response_latency_n_overlapping": 0,
            "response_latency_prop_positive": np.nan,
            "response_latency_prop_near_zero": np.nan,
            "response_latency_prop_overlapping": np.nan,
        }, []

    arr = np.array(latencies, dtype=float)
    n = len(arr)
    n_pos = int(np.sum(arr > NEAR_ZERO_GAP_TOLERANCE_S))
    n_near_zero = int(np.sum(np.abs(arr) <= NEAR_ZERO_GAP_TOLERANCE_S))
    n_overlap = int(np.sum(arr < -NEAR_ZERO_GAP_TOLERANCE_S))
    stats = {
        "response_latency_mean": float(np.mean(arr)),
        "response_latency_median": float(np.median(arr)),
        "response_latency_std": float(np.std(arr)),
        "response_latency_min": float(np.min(arr)),
        "response_latency_max": float(np.max(arr)),
        "response_latency_n": n,
        "response_latency_n_positive": n_pos,
        "response_latency_n_near_zero": n_near_zero,
        "response_latency_n_overlapping": n_overlap,
        "response_latency_prop_positive": n_pos / n,
        "response_latency_prop_near_zero": n_near_zero / n,
        "response_latency_prop_overlapping": n_overlap / n,
    }
    return stats, latencies


def _find_low_energy_runs(times: np.ndarray, low_mask: np.ndarray) -> list:
    """Contiguous runs of True in low_mask -> (start_time, end_time) pairs."""
    idx = np.where(low_mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = []
    for group in np.split(idx, splits):
        start = float(times[group[0]])
        end = float(times[group[-1]] + FRAME_HOP_S)
        runs.append((start, end))
    return runs


def detect_within_turn_pauses(target_turns: list, times: np.ndarray, rms: np.ndarray) -> list:
    """
    Candidate ACOUSTIC pauses inside target turns only — never linguistic
    "hesitations" (that label would need NLP/prosodic evidence this stage
    doesn't have). `times`/`rms` are this recording's ORIGINAL (not
    denoised) Channel-0 time-series from outputs/features/timeseries.csv.

    Threshold: PAUSE_RMS_PERCENTILE-th percentile of the RMS values found
    strictly *inside this recording's own target turns* (see module
    docstring). Minimum duration: MIN_PAUSE_DURATION_S, to avoid flagging
    single quiet frames (e.g. between syllables) as pauses.
    """
    if len(times) == 0 or not target_turns:
        return []

    order = np.argsort(times)
    times, rms = times[order], rms[order]

    in_turn_mask = np.zeros(len(times), dtype=bool)
    for t in target_turns:
        in_turn_mask |= (times >= t["start"]) & (times < t["end"])
    in_turn_rms = rms[in_turn_mask]
    if len(in_turn_rms) == 0:
        return []
    threshold = float(np.percentile(in_turn_rms, PAUSE_RMS_PERCENTILE))

    pauses = []
    for t in target_turns:
        mask = (times >= t["start"]) & (times < t["end"])
        turn_times = times[mask]
        turn_rms = rms[mask]
        if len(turn_times) == 0:
            continue
        for start, end in _find_low_energy_runs(turn_times, turn_rms < threshold):
            end = min(end, t["end"])  # stay strictly inside the turn
            duration = end - start
            if duration >= MIN_PAUSE_DURATION_S:
                pauses.append({
                    "turn_start": t["start"],
                    "turn_end": t["end"],
                    "pause_start": start,
                    "pause_end": end,
                    "pause_duration": duration,
                })
    return pauses


def within_turn_pause_summary(pauses: list, target_total_speaking_time: float) -> dict:
    if not pauses:
        return {
            "within_turn_pause_count": 0,
            "within_turn_pause_total_duration": 0.0,
            "within_turn_pause_mean_duration": np.nan,
            "within_turn_pause_median_duration": np.nan,
            "within_turn_pause_max_duration": np.nan,
            "within_turn_pause_ratio": (0.0 if target_total_speaking_time else np.nan),
        }
    durations = np.array([p["pause_duration"] for p in pauses])
    total = float(np.sum(durations))
    return {
        "within_turn_pause_count": len(pauses),
        "within_turn_pause_total_duration": total,
        "within_turn_pause_mean_duration": float(np.mean(durations)),
        "within_turn_pause_median_duration": float(np.median(durations)),
        "within_turn_pause_max_duration": float(np.max(durations)),
        "within_turn_pause_ratio": (total / target_total_speaking_time) if target_total_speaking_time else np.nan,
    }


def _cv(values) -> float:
    """Coefficient of variation = std / |mean|. NaN if empty or mean is 0
    (a near-zero-mean quantity like response latency makes CV unstable —
    callers/readers should treat a CV near a zero-mean feature with care)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0 or np.mean(values) == 0:
        return np.nan
    return float(np.std(values) / abs(np.mean(values)))


def _quantiles(values, prefix: str, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    out = {}
    for q in qs:
        key = f"{prefix}_q{int(round(q * 100))}"
        out[key] = float(np.percentile(values, q * 100)) if len(values) else np.nan
    return out


def variability_stats(target_turns: list, response_latencies: list, gaps: list) -> dict:
    target_durations = [t["end"] - t["start"] for t in target_turns]
    stats = {
        "target_turn_duration_cv": _cv(target_durations),
        "response_latency_cv": _cv(response_latencies),
        "inter_turn_gap_cv": _cv(gaps),
    }
    stats.update(_quantiles(target_durations, "target_turn_duration"))
    stats.update(_quantiles(response_latencies, "response_latency"))
    stats.update(_quantiles(gaps, "inter_turn_gap"))

    td = np.asarray(target_durations, dtype=float)
    stats["n_short_target_turns"] = int(np.sum(td < SHORT_TURN_THRESHOLD_S)) if len(td) else 0
    stats["n_long_target_turns"] = int(np.sum(td > LONG_TURN_THRESHOLD_S)) if len(td) else 0
    return stats
