"""
Optional noise-reduction path for Channel 0.

Deliberately simple and interpretable per rules.md: a single high-pass
Butterworth filter that removes low-frequency room noise (hum, rumble,
AC/fan noise) below `CUTOFF_HZ`. It does NOT touch the frequency range
where speech energy and formants live, so it is a mild, conservative
choice rather than an aggressive black-box denoiser.

This never overwrites the original signal — callers keep both arrays
and run feature extraction on each separately.
"""
from scipy.signal import butter, filtfilt

METHOD = "butterworth_highpass"
CUTOFF_HZ = 80.0
ORDER = 4


def denoise_channel(y, sr):
    """
    Zero-phase high-pass filter (filtfilt avoids the time-shift a
    single-pass IIR filter would introduce, which matters because we
    later align features by frame time).

    Returns (denoised_signal, params_used).
    """
    nyquist = sr / 2.0
    normalized_cutoff = CUTOFF_HZ / nyquist
    b, a = butter(ORDER, normalized_cutoff, btype="highpass")
    denoised = filtfilt(b, a, y)

    params = {
        "method": METHOD,
        "cutoff_hz": CUTOFF_HZ,
        "order": ORDER,
    }
    return denoised, params
