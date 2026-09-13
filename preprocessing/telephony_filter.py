"""
Telephony-band filter — a NEW, additive experiment path testing whether
the human-vs-synthetic separation survives a common, plausible
telephone-channel frequency constraint. Does not touch or modify
preprocessing/denoise.py (a high-pass only) or preprocessing/normalize.py
(a pure gain) — this is a third, independent signal-processing path.

Filter design (identical for every recording, independent of label):
  - Band-pass Butterworth, order 4, cutoffs 300 Hz - 3400 Hz
    (the classic "plain old telephone service" analog voice-band —
    not tuned to this dataset in any way, just a standard, conservative
    telephony bandwidth reference).
  - Applied zero-phase (scipy.signal.sosfiltfilt) so it introduces no
    time delay/phase distortion that could shift frame alignment
    relative to the existing turn timestamps.
  - Second-order-sections (sos) form used instead of transposed-direct-
    form (b, a) coefficients for numerical stability at this filter
    order/bandwidth combination.
  - No resampling (already 8000 Hz; Nyquist 4000 Hz, so 3400 Hz is a
    valid, non-degenerate upper cutoff at 0.85 of Nyquist).
  - No denoising, no compression, no EQ beyond this single band limit,
    no other normalization in this module — RMS normalization, where
    used, is applied as a SEPARATE, explicit second step by the caller
    (see preprocessing/telephony_pipeline.py variant B), never fused
    into this function.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

LOW_CUTOFF_HZ = 300.0
HIGH_CUTOFF_HZ = 3400.0
ORDER = 4
METHOD = "butterworth_bandpass_sos"


def apply_telephony_band(y: np.ndarray, sr: int) -> tuple:
    nyquist = sr / 2.0
    low = LOW_CUTOFF_HZ / nyquist
    high = HIGH_CUTOFF_HZ / nyquist
    sos = butter(ORDER, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, y)
    params = {
        "method": METHOD,
        "low_cutoff_hz": LOW_CUTOFF_HZ,
        "high_cutoff_hz": HIGH_CUTOFF_HZ,
        "order": ORDER,
    }
    return filtered, params
