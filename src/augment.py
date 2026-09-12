"""
Augmentation telefonica para hacer el modelo robusto a:
  - Ruido de fondo (linea, ambiente)
  - Filtrado de banda telefonica (300-3400 Hz)
  - Pequenas variaciones de ganancia
"""
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_telephony(y, sr, low=300, high=3400):
    """Aplica filtro de banda telefonica."""
    nyq = sr / 2.0
    high = min(high, nyq * 0.95)
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, y)


def add_noise(y, snr_db=20):
    """Anade ruido blanco con SNR controlado."""
    sig_power = np.mean(y ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise


def random_gain(y, db_range=(-3, 3)):
    """Aplica ganancia aleatoria."""
    gain_db = np.random.uniform(*db_range)
    return y * (10 ** (gain_db / 20))


def augment_telephony(y, sr, seed=None):
    """Pipeline completo de augmentation. Aplica con probabilidad."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    out = y.copy()

    # Filtro telefonico (siempre)
    out = bandpass_telephony(out, sr)

    # Ruido con prob 0.7
    if rng.random() < 0.7:
        snr = rng.uniform(15, 30)
        out = add_noise(out, snr_db=snr)

    # Ganancia con prob 0.5
    if rng.random() < 0.5:
        out = random_gain(out, db_range=(-3, 3))

    # Normalizar para que el VAD funcione
    max_abs = np.max(np.abs(out)) + 1e-12
    if max_abs > 0.99:
        out = out / max_abs * 0.95

    return out
