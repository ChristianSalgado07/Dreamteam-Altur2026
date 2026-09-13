"""
Per-session evidence log + "AI Voice Analysis Report" for the live
detector. Answers "what exactly is the detector flagging?" using two
strictly separated kinds of evidence:

  1. MODEL EVIDENCE -- what the CNN actually predicted, per speech
     window (the primary, only real-time classifier evidence).
  2. ACOUSTIC OBSERVATIONS -- descriptive statistics of the SAME
     features this project's earlier research already computed
     (preprocessing/features.py), for this session only. These are
     reported as measurements, not as independent proof of anything.

Whether a feature family gets called "previously found diagnostic" is
NOT invented here -- it is looked up from this project's own prior,
already-published univariate separability analysis
(outputs/features/modeling_univariate_separability.csv,
outputs/reports/modeling_evaluation_report.md), which found:
  - std_zcr, spectral rolloff/bandwidth/centroid range, and a few MFCC
    coefficients (esp. mfcc8) as the strongest univariate predictors
    (AUC 0.90-0.98) on the ORIGINAL dataset.
  - raw_rms/peak amplitude ALSO scored moderately high (AUC ~0.81-0.85)
    but that report explicitly flagged this as a likely
    loudness-normalization / recording-pipeline confound, not a
    genuine speech characteristic -- so this module deliberately does
    NOT present RMS/peak-amplitude consistency as independent AI
    evidence, regardless of what it measures in a given session.
  - F0/pitch (range_f0/max_f0) scored moderately (AUC ~0.78) -- now
    computed live (needed for the speaker-change guard, see
    app/speaker_guard.py) and surfaced here when available; still
    marked inconclusive for windows/sessions where too few voiced
    frames were found to trust it.

This module never re-derives new AUC numbers at runtime -- it only
cites the existing, already-reviewed research findings above and
reports fresh descriptive statistics (mean / coefficient of variation)
for THIS session's own speech windows.

SPEAKER SEGMENTS: the session is now split into consecutive per-speaker
segments (app/speaker_guard.py decides when a new one starts). Each
speaker gets its own independent evidence tally -- a speaker change
never erases a prior speaker's report, it starts a new one (see
summary()'s "speaker_segments" and the timeline's "speaker_change" /
"stabilizing" entries).
"""
import numpy as np

from preprocessing.audio_io import basic_signal_stats
from preprocessing.features import extract_rms, extract_spectral, extract_mfcc, mfcc_summary, extract_f0
from app.audio_utils import WINDOW_SECONDS, HOP_SECONDS
from app.decision import AI_THRESHOLD, HUMAN_THRESHOLD
from app.speaker_guard import MIN_VOICED_FRAMES


def compute_window_f0(raw_buffer: np.ndarray, sr: int):
    """
    Median F0 over voiced pYIN frames in this window, or None if too few
    frames were judged voiced to trust the estimate (common in this
    project's 8kHz telephone audio -- empirically ~30% of speech
    windows). Reuses the existing extract_f0 (preprocessing/features.py)
    unchanged. Used both for the speaker-change guard and the acoustic
    evidence report's pitch row.
    """
    _, f0, voiced_flag, _ = extract_f0(raw_buffer, sr)
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < MIN_VOICED_FRAMES:
        return None
    return float(np.median(voiced))

# Context labels sourced from outputs/reports/modeling_evaluation_report.md
# and outputs/features/modeling_univariate_separability.csv -- see module
# docstring. Not computed or fitted here; this is a fixed reference table.
FEATURE_RESEARCH_CONTEXT = {
    "spectral_centroid": ("diagnostic", "previously found diagnostic in this project's own dataset (range/max AUC ~0.91)"),
    "zero_crossing_rate": ("diagnostic", "the single strongest univariate feature in this project's own dataset (AUC ~0.98)"),
    "spectral_bandwidth": ("diagnostic", "previously found diagnostic in this project's own dataset (range/max AUC ~0.93)"),
    "spectral_rolloff": ("diagnostic", "previously found diagnostic in this project's own dataset (range/max AUC ~0.98)"),
    "mfcc_pattern": ("diagnostic", "coefficient 8 in particular was previously found diagnostic (AUC ~0.93); not independently re-verified this session"),
    "rms_energy": ("confound", "scored moderately high in prior research (AUC ~0.81-0.85) but was flagged there as a likely recording-loudness confound, not a speech characteristic -- not treated as independent AI evidence here"),
    "peak_amplitude": ("confound", "same recording-loudness confound noted for RMS energy -- not treated as independent AI evidence here"),
    "pitch_prosody": ("diagnostic", "previously found moderately diagnostic in this project's own dataset (range/max F0 AUC ~0.78); only available for windows with enough voiced frames"),
}


def compute_window_features(raw_buffer: np.ndarray, norm_buffer: np.ndarray, sr: int) -> dict:
    """
    Descriptive acoustic features for one already-scored window, reusing
    the project's existing extractors (preprocessing/features.py) --
    no new feature-extraction logic. RMS/peak/ZCR/spectral shape are
    computed on the RAW (pre-normalization) buffer so they reflect the
    actual recording, not an artifact of this app's own per-window RMS
    normalization; MFCCs are computed on the normalized buffer (the same
    signal version the CNN itself sees).
    """
    stats = basic_signal_stats(raw_buffer, sr)
    rms_frames = extract_rms(raw_buffer)
    centroid, bandwidth, rolloff, zcr = extract_spectral(raw_buffer, sr)
    mfcc = extract_mfcc(norm_buffer, sr)
    mfcc_means = {k: v for k, v in mfcc_summary(mfcc).items() if k.endswith("_mean")}

    return {
        "peak_amplitude": stats["peak_amplitude"],
        "rms": stats["rms"],
        "zcr": float(np.mean(zcr)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "mfcc_means": mfcc_means,
    }


def _classify_raw(raw: float) -> str:
    if raw >= AI_THRESHOLD:
        return "AI-like"
    if raw <= HUMAN_THRESHOLD:
        return "Human-like"
    return "Uncertain"


def _variation_label(cv: float) -> str:
    """Purely descriptive of THIS session's own measurements -- not a
    diagnostic/AI-relevance claim (see FEATURE_RESEARCH_CONTEXT for that)."""
    if cv < 0.10:
        return "Consistent"
    if cv < 0.25:
        return "Moderate variation"
    return "High variation"


def _speaker_result(speech: list, current_label: str = None) -> dict:
    """Shared per-speaker tally logic, used both for a finished/prior
    speaker segment and (with current_label) the active one."""
    n = len(speech)
    if n < 2:
        return {"status": "Analyzing...", "overall_result": None, "evidence_strength": None,
                "n_windows": n, "ai_count": 0, "human_count": 0, "uncertain_count": n}

    ai_count = sum(1 for s in speech if s["window_classification"] == "AI-like")
    human_count = sum(1 for s in speech if s["window_classification"] == "Human-like")
    uncertain_count = n - ai_count - human_count
    majority_frac = max(ai_count, human_count) / n

    if current_label is None:
        # Finished segment: no live hysteresis state -- derive a result
        # directly from the majority of its own windows (same, existing
        # AI/Human thresholds, just applied to counts instead of a
        # smoothed running score).
        if ai_count > human_count and ai_count >= uncertain_count:
            result = "AI likely"
        elif human_count > ai_count and human_count >= uncertain_count:
            result = "Human likely"
        else:
            result = "Uncertain"
    else:
        result = current_label

    if result in ("AI likely", "Human likely"):
        strength = "STRONG" if majority_frac >= 0.8 else "MODERATE" if majority_frac >= 0.6 else "WEAK"
    else:
        strength = "WEAK"

    return {"status": None, "overall_result": result, "evidence_strength": strength,
            "n_windows": n, "ai_count": ai_count, "human_count": human_count, "uncertain_count": uncertain_count}


class EvidenceSession:
    """
    Ordered log of every window analyzed since START (or since a file
    simulation began), used to render the AI Voice Analysis Report.
    Timestamps are derived from a deterministic window index
    (index * HOP_SECONDS), never from wall-clock time, so repeated runs
    of the same file produce an identical report.

    The session is split into speaker segments (see app/speaker_guard.py
    for WHEN a new one starts); each keeps its own independent list of
    speech windows, so a speaker change never erases a prior speaker's
    evidence.
    """

    def __init__(self):
        self.speakers = [[]]     # list of speaker segments, each a list of speech-window dicts
        self.timeline = []       # global ordered log: speech/silence/stabilizing/speaker_change
        self.window_index = 0
        self.speaker_changes = 0

    def next_t_start(self) -> float:
        return self.window_index * HOP_SECONDS

    def add_silence(self, t_start: float):
        self.timeline.append({"type": "silence", "t_start": t_start, "t_end": t_start + WINDOW_SECONDS})
        self.window_index += 1

    def add_stabilizing(self, t_start: float):
        self.timeline.append({"type": "stabilizing", "t_start": t_start, "t_end": t_start + WINDOW_SECONDS})
        self.window_index += 1

    def add_speaker_change(self, t_start: float):
        self.speaker_changes += 1
        self.timeline.append({"type": "speaker_change", "t_start": t_start, "t_end": t_start})
        self.speakers.append([])
        self.window_index += 1

    def add_speech(self, t_start: float, raw: float, smoothed: float, label: str, features: dict, f0_median):
        record = {
            "t_start": t_start, "t_end": t_start + WINDOW_SECONDS,
            "raw": raw, "smoothed": smoothed, "label": label,
            "window_classification": _classify_raw(raw),
            "features": features, "f0_median": f0_median,
        }
        self.speakers[-1].append(record)
        self.timeline.append({
            "type": "speech", "t_start": t_start, "t_end": t_start + WINDOW_SECONDS,
            "label": record["window_classification"], "speaker_index": len(self.speakers),
        })
        self.window_index += 1

    def summary(self, current_label: str) -> dict:
        current_speech = self.speakers[-1]

        speaker_segments = []
        for i, seg in enumerate(self.speakers):
            is_current = (i == len(self.speakers) - 1)
            r = _speaker_result(seg, current_label if is_current else None)
            t_start = seg[0]["t_start"] if seg else None
            t_end = seg[-1]["t_end"] if seg else None
            speaker_segments.append({"speaker_index": i + 1, "t_start": t_start, "t_end": t_end, **r})

        n = len(current_speech)
        if n == 0:
            return {
                "overall_result": "No speech analyzed yet" if len(self.speakers) == 1 else "Analyzing...",
                "evidence_strength": None,
                "n_windows": 0, "speech_seconds": 0.0,
                "ai_count": 0, "human_count": 0, "uncertain_count": 0,
                "flags": [], "not_enough_evidence": [
                    "No speech windows analyzed yet for this speaker -- start speaking or run a file simulation.",
                ],
                "acoustic_summary": [],
                "strongest_windows": [], "weakest_windows": [],
                "speaker_segments": speaker_segments, "speaker_changes": self.speaker_changes,
                "timeline": list(self.timeline),
            }

        r = _speaker_result(current_speech, current_label)
        ai_count, human_count, uncertain_count = r["ai_count"], r["human_count"], r["uncertain_count"]
        strength = r["evidence_strength"]

        flags = []
        not_enough = []
        if ai_count >= max(2, int(0.6 * n)):
            flags.append("Repeated AI-like spectral patterns across independent speech windows")
        if n >= 2 and ai_count == n:
            flags.append("Consistent evidence across every analyzed speech window")
        if current_label == "AI likely":
            flags.append(f"High model scores across {ai_count}/{n} analyzed windows")
        if current_label == "Human likely":
            flags.append(f"Consistently low model (AI-likelihood) scores across {human_count}/{n} analyzed windows")
        if not flags:
            not_enough.append("Model scores across windows were inconsistent -- no repeated pattern to flag")
        n_f0 = sum(1 for s in current_speech if s.get("f0_median") is not None)
        if n_f0 == 0:
            not_enough.append("Pitch/prosody evidence: inconclusive (no window had enough voiced frames)")
        not_enough.append("Silence/pauses: not used as evidence in either direction")
        not_enough.append("Speaker identity: not verified -- speaker-change guard only detects a sustained acoustic shift, it does not identify who is speaking")

        acoustic_summary = self._acoustic_summary(current_speech)
        raws = [(s["t_start"], s["t_end"], s["raw"]) for s in current_speech]
        strongest = sorted(raws, key=lambda r: -r[2])[:4]
        weakest = sorted(raws, key=lambda r: r[2])[:2]

        return {
            "overall_result": current_label,
            "evidence_strength": strength,
            "n_windows": n,
            "speech_seconds": round(n * WINDOW_SECONDS, 1),
            "ai_count": ai_count, "human_count": human_count, "uncertain_count": uncertain_count,
            "flags": flags, "not_enough_evidence": not_enough,
            "acoustic_summary": acoustic_summary,
            "strongest_windows": [{"t_start": a, "t_end": b, "score": c} for a, b, c in strongest],
            "weakest_windows": [{"t_start": a, "t_end": b, "score": c} for a, b, c in weakest],
            "speaker_segments": speaker_segments, "speaker_changes": self.speaker_changes,
            "timeline": list(self.timeline),
        }

    def _acoustic_summary(self, speech: list) -> list:
        rows = []
        feature_keys = [
            ("spectral_centroid", "Spectral centroid", "Hz"),
            ("zcr", "Zero-crossing rate", ""),
            ("rms", "RMS energy", ""),
            ("spectral_bandwidth", "Spectral bandwidth", "Hz"),
            ("spectral_rolloff", "Spectral rolloff", "Hz"),
        ]
        for key, label, unit in feature_keys:
            values = np.array([s["features"][key] for s in speech if key in s.get("features", {})])
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values))
            cv = (std / mean) if mean else 0.0
            context_key = {
                "spectral_centroid": "spectral_centroid", "zcr": "zero_crossing_rate",
                "rms": "rms_energy", "spectral_bandwidth": "spectral_bandwidth",
                "spectral_rolloff": "spectral_rolloff",
            }[key]
            kind, note = FEATURE_RESEARCH_CONTEXT[context_key]
            diagnostic_label = "Measured -- not independently diagnostic" if kind != "diagnostic" else "AI-associated pattern (prior research)"
            rows.append({
                "feature": label, "unit": unit, "mean": mean, "std": std,
                "variation_label": _variation_label(cv),
                "diagnostic_label": diagnostic_label, "research_note": note,
            })
        # MFCC pattern -- single combined row, mean of coefficient means
        mfcc_all = [s["features"].get("mfcc_means", {}) for s in speech]
        if mfcc_all and mfcc_all[0]:
            keys = list(mfcc_all[0].keys())
            matrix = np.array([[m.get(k, np.nan) for k in keys] for m in mfcc_all])
            mean_vec = np.nanmean(matrix, axis=0)
            std_vec = np.nanstd(matrix, axis=0)
            cv = float(np.nanmean(std_vec / np.where(mean_vec == 0, np.nan, mean_vec)))
            kind, note = FEATURE_RESEARCH_CONTEXT["mfcc_pattern"]
            rows.append({
                "feature": "MFCC pattern", "unit": "", "mean": float(np.nanmean(mean_vec)), "std": float(np.nanmean(std_vec)),
                "variation_label": _variation_label(abs(cv) if cv == cv else 0.0),
                "diagnostic_label": "AI-associated pattern (prior research)", "research_note": note,
            })
        # Pitch/F0 -- only from windows where it was actually available
        f0_values = np.array([s["f0_median"] for s in speech if s.get("f0_median") is not None])
        kind, note = FEATURE_RESEARCH_CONTEXT["pitch_prosody"]
        if len(f0_values) >= 2:
            mean = float(np.mean(f0_values))
            cv = float(np.std(f0_values) / mean) if mean else 0.0
            rows.append({
                "feature": "Pitch (F0)", "unit": "Hz", "mean": mean, "std": float(np.std(f0_values)),
                "variation_label": _variation_label(cv),
                "diagnostic_label": "AI-associated pattern (prior research)",
                "research_note": note + f" ({len(f0_values)}/{len(speech)} windows had usable pitch)",
            })
        else:
            rows.append({
                "feature": "Pitch / prosody", "unit": "", "mean": None, "std": None,
                "variation_label": "Not computed", "diagnostic_label": "Inconclusive",
                "research_note": "too few windows had enough voiced frames to trust a pitch estimate this session",
            })
        return rows
