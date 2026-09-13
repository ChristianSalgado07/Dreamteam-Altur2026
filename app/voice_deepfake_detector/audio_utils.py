"""
audio_utils.py - Pipeline final V2 (61 features).

Combina:
  - 39 MFCCs (canal 0, normalizado)
  - 10 features acústicas extra (ZCR, centroid, bandwidth, rolloff, RMS)
  - 12 features de comportamiento (VAD, turnos, interrupciones, latencias)
"""
import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import medfilt
from tqdm import tqdm
from config import AUDIO_DIR, SAMPLE_RATE, N_MFCC

"""Carga un audio por anon_id. Devuelve caller (canal 0) por defecto.
   Utilizada para una versión anterior del pipeline"""
def cargar_audio(anon_id, sr=SAMPLE_RATE, canal=0, normalizar=True):
    ruta = os.path.join(AUDIO_DIR, f"{anon_id}.wav")
    if canal is None:
        y, _ = librosa.load(ruta, sr=sr, mono=True)
    else:
        y, _ = librosa.load(ruta, sr=sr, mono=False)
        if y.ndim == 2:
            y = y[canal]
    if normalizar and len(y) > 0:
        y = librosa.util.normalize(y)
    return y

"""Carga caller y agente.
   Se normaliza el canal escalandose a amplitud [-1,1]."""
def cargar_ambos_canales(anon_id, sr=SAMPLE_RATE, normalizar=True):
    ruta = os.path.join(AUDIO_DIR, f"{anon_id}.wav")
    y, _ = librosa.load(ruta, sr=sr, mono=False)
    if y.ndim == 1:
        caller, agente = y, np.zeros_like(y)
    else:
        caller = y[0]
        agente = y[1] if y.shape[0] > 1 else np.zeros_like(caller)
    if normalizar:
        if len(caller) > 0:
            caller = librosa.util.normalize(caller)
        if len(agente) > 0:
            agente = librosa.util.normalize(agente)
    return caller, agente

""" Mel-Frequency Cepstral Coefficient. Cuánta energía hay en 
    una banda de frecuencia específica (escala Mel).
    Media = timbre global, desv est = variación del timbre en tiempo
    y delta = rapidez con la que cambia el timbre"""
def _mfcc_features(y, sr, n_mfcc=N_MFCC):
    if len(y) == 0:
        return np.zeros(3 * n_mfcc)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([
        np.mean(mfcc, axis=1), # 13 medias
        np.std(mfcc, axis=1), # 13 desviaciones
        np.mean(librosa.feature.delta(mfcc), axis=1), # 13 deltas
    ])

""" Features acústicas extra, promedios y desv est.
    Zcr = cuántas veces la señal cruza el eje cero.
    Centroid = mide las frecuencias.
    Bandwidth = varianza del espectro.
    Rolloff = frecuencia bajo la cual está el 85% de la energía.
    RMS = volumen. """
def _acoustic_extra_features(y, sr):
    if len(y) == 0:
        return np.zeros(10)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    return np.array([
        np.mean(zcr), np.std(zcr),
        np.mean(centroid), np.std(centroid),
        np.mean(bandwidth), np.std(bandwidth),
        np.mean(rolloff), np.std(rolloff),
        np.mean(rms), np.std(rms),
    ])

""" 
    Divide el audio en frames de 20ms, calcula la energía RMS de cada frame,
    marca como voz los frames con energía > umbral, suaviza el medfilt para
    eliminar ruido.
    Retorna un array booleano para marcar donde hay voz(T) y donde silencio(F)
"""
def _get_vad_flags(y, sr, frame_ms=20, energy_threshold=None):
    frame_size = int(sr * frame_ms / 1000)
    if len(y) < frame_size:
        return np.array([], dtype=bool)
    n_frames = len(y) // frame_size
    if n_frames == 0:
        return np.array([], dtype=bool)
    trimmed = y[:n_frames * frame_size].reshape(n_frames, frame_size)
    energies = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
    if energy_threshold is None:
        energy_threshold = max(np.percentile(energies, 30) * 1.5, 1e-4)
    flags = (energies > energy_threshold).astype(float)
    flags = medfilt(flags, kernel_size=5).astype(bool)
    return flags

""" Encuentra transiciones entre voz y silencio.
    Filtra "turnos" de menos de 60 ms (ruido) """
def _vad_to_segments(flags, frame_ms=20):
    if len(flags) == 0:
        return []
    padded = np.concatenate([[False], flags, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    min_frames = int(60 / frame_ms)
    mask = (ends - starts) >= min_frames
    return list(zip(starts[mask], ends[mask]))

""" Comportamiento del caller y agente, tiempo de habla, interrupciones, 
    turnos, overlaps, etc."""
def _behavioral_features(caller, agente, sr):
    frame_ms = 20
    caller_flags = _get_vad_flags(caller, sr, frame_ms)
    agent_flags = _get_vad_flags(agente, sr, frame_ms)
    if len(caller_flags) == 0 or len(agent_flags) == 0:
        return np.zeros(12)
    min_len = min(len(caller_flags), len(agent_flags))
    caller_flags = caller_flags[:min_len]
    agent_flags = agent_flags[:min_len]
    caller_segs = _vad_to_segments(caller_flags, frame_ms)
    agent_segs = _vad_to_segments(agent_flags, frame_ms)
    total_time = min_len * frame_ms / 1000.0
    caller_speech = np.sum(caller_flags) * frame_ms / 1000.0
    agent_speech = np.sum(agent_flags) * frame_ms / 1000.0
    overlap = np.sum(caller_flags & agent_flags) * frame_ms / 1000.0
    caller_turns = len(caller_segs)
    agent_turns = len(agent_segs)
    caller_durs = [(e - s) * frame_ms / 1000.0 for s, e in caller_segs]
    avg_caller_turn = np.mean(caller_durs) if caller_durs else 0
    std_caller_turn = np.std(caller_durs) if caller_durs else 0
    latencies = []
    if caller_segs and agent_segs:
        caller_starts = np.array([s for s, _ in caller_segs])
        agent_ends = np.array([e for _, e in agent_segs])
        for c_start in caller_starts:
            valid = agent_ends[agent_ends <= c_start]
            if len(valid) > 0:
                latencies.append((c_start - valid.max()) * frame_ms / 1000.0)
    avg_latency = np.mean(latencies) if latencies else 0
    std_latency = np.std(latencies) if latencies else 0
    caller_rising = np.concatenate([[False], caller_flags[:-1] == False]) & caller_flags
    agent_rising = np.concatenate([[False], agent_flags[:-1] == False]) & agent_flags
    caller_interr = int(np.sum(caller_rising & agent_flags))
    agent_interr = int(np.sum(agent_rising & caller_flags))
    return np.array([
        caller_speech / total_time if total_time > 0 else 0,
        agent_speech / total_time if total_time > 0 else 0,
        overlap / total_time if total_time > 0 else 0,
        caller_turns,
        agent_turns,
        avg_caller_turn,
        std_caller_turn,
        avg_latency,
        std_latency,
        caller_interr,
        agent_interr,
        total_time,
    ])

""" El modelo esta entrenado con los features en este orden"""
def extraer_features_v2(caller, agente, sr=SAMPLE_RATE):
    """61 features por audio."""
    base = _mfcc_features(caller, sr) # 39
    extra = _acoustic_extra_features(caller, sr) # 10
    behavior = _behavioral_features(caller, agente, sr) # 12
    return np.concatenate([base, extra, behavior]) # 61

""" Compatibilidad: recibe solo caller y rellena agente vacío. """
def extraer_features(y, sr=SAMPLE_RATE):
    return extraer_features_v2(y, np.zeros_like(y), sr)


def extraer_features_desde_archivo(anon_id):
    caller, agente = cargar_ambos_canales(anon_id)
    return extraer_features_v2(caller, agente)

""" Itera el manifest y extrae 61 features por audio. """
def extraer_features_dataset(manifest_path, verbose=True):
    df = pd.read_csv(manifest_path)
    X, y, ids = [], [], []
    iterador = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
    for _, row in iterador:
        try:
            feats = extraer_features_desde_archivo(row["anon_id"])
            X.append(feats)
            y.append(row["label"])
            ids.append(row["anon_id"])
        except Exception as e:
            print(f"[ERROR] {row['anon_id']}: {e}")
    return np.array(X), np.array(y), ids


def nombres_features(n_mfcc=N_MFCC):
    base = (
        [f"MFCC_{i+1}_mean" for i in range(n_mfcc)]
        + [f"MFCC_{i+1}_std" for i in range(n_mfcc)]
        + [f"MFCC_{i+1}_delta" for i in range(n_mfcc)]
    )
    extra = [
        "zcr_mean", "zcr_std",
        "centroid_mean", "centroid_std",
        "bandwidth_mean", "bandwidth_std",
        "rolloff_mean", "rolloff_std",
        "rms_mean", "rms_std",
    ]
    behavior = [
        "caller_speech_ratio", "agent_speech_ratio", "overlap_ratio",
        "caller_turns", "agent_turns",
        "avg_caller_turn", "std_caller_turn",
        "avg_latency", "std_latency",
        "caller_interruptions", "agent_interruptions",
        "total_time",
    ]
    return base + extra + behavior