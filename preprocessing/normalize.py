"""
RMS-based level normalization — a NEW, additive preprocessing path for
the level-normalization confound experiment (see
outputs/reports/normalization_experiment_report.md). Does not replace
or modify the existing original/denoised paths (preprocessing/denoise.py
is untouched) — this is a third, parallel signal version.

Formula (a single linear gain per recording, computed from that
recording's own samples only):

    gain = min(TARGET_RMS / current_rms, PEAK_SAFETY / peak_amplitude)
    y_normalized = y * gain

- `TARGET_RMS = 0.05`: fixed, label-independent constant. Chosen as a
  round number close to (but not fitted to) this dataset's pooled
  natural RMS level — human recordings average raw_rms≈0.044, synthetic
  ≈0.078 (see the original diagnostics), so 0.05 sits between them
  rather than favoring either group. It is NOT tuned per class and
  never uses the label.
- `PEAK_SAFETY = 0.99`: if scaling to TARGET_RMS would push a
  recording's peak sample above this, the gain is capped instead so the
  signal never digitally clips. This is still a single constant
  multiplier for the whole file — not compression, limiting, or any
  time-varying process — just the more conservative of two possible
  constant gains.
- Each recording is normalized using ONLY its own RMS/peak — never the
  label, never another recording's statistics, never a class-level
  target.
- No resampling, no channel-count change, no denoising, no EQ.
"""
import numpy as np

TARGET_RMS = 0.05
PEAK_SAFETY = 0.99
MIN_RMS_FLOOR = 1e-6  # guards against a degenerate (near-silent) signal


def normalize_rms(y: np.ndarray) -> tuple:
    """Returns (normalized_signal, info_dict). info_dict documents
    exactly what happened for transparency/reporting — never silent."""
    current_rms = float(np.sqrt(np.mean(np.square(y))))
    peak_amplitude = float(np.max(np.abs(y))) if len(y) else 0.0

    if current_rms < MIN_RMS_FLOOR:
        return y.copy(), {
            "current_rms": current_rms, "peak_amplitude": peak_amplitude,
            "gain": 1.0, "peak_capped": False, "degenerate_signal": True,
        }

    gain_for_target = TARGET_RMS / current_rms
    gain_cap = (PEAK_SAFETY / peak_amplitude) if peak_amplitude > 0 else gain_for_target
    gain = min(gain_for_target, gain_cap)
    peak_capped = gain < gain_for_target

    normalized = (y * gain).astype(y.dtype if np.issubdtype(y.dtype, np.floating) else np.float64)
    return normalized, {
        "current_rms": current_rms, "peak_amplitude": peak_amplitude,
        "gain": gain, "peak_capped": peak_capped, "degenerate_signal": False,
    }
