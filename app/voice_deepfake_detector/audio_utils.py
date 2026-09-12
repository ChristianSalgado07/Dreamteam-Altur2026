"""
Funciones para cargar audio y extraer características.

Cambios respecto a la versión anterior:
  - Se carga SOLO el canal 0 (caller). El canal 1 es el agente.
  - Se aplica normalización de amplitud como buena práctica.
"""
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from config import AUDIO_DIR, SAMPLE_RATE, N_MFCC


def cargar_audio(anon_id, sr=SAMPLE_RATE, canal=0, normalizar=True):
    """
    Carga un audio desde su anon_id.

    Args:
        anon_id: identificador del audio
        sr: sample rate objetivo
        canal: 0 = caller (default), 1 = agente, None = mono (promedio)
        normalizar: si True, escala la amplitud entre -1 y 1

    Returns:
        array 1D con el audio procesado
    """
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


def extraer_features(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Extrae 3*n_mfcc características de un audio:
      - Media de MFCCs
      - Desviación estándar de MFCCs
      - Media de Delta MFCCs
    Devuelve un vector de 39 valores (con n_mfcc=13).
    """
    if len(y) == 0:
        return np.zeros(3 * n_mfcc)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta_mean = np.mean(librosa.feature.delta(mfcc), axis=1)
    return np.concatenate([mfcc_mean, mfcc_std, mfcc_delta_mean])


def extraer_features_dataset(manifest_path, verbose=True):
    """
    Itera sobre el manifest y extrae features de todos los audios.
    Usa canal 0 + normalización (el pipeline final elegido).
    """
    df = pd.read_csv(manifest_path)
    X, y, ids = [], [], []

    iterador = tqdm(df.iterrows(), total=len(df)) if verbose else df.iterrows()
    for _, row in iterador:
        anon_id = row["anon_id"]
        try:
            audio = cargar_audio(anon_id, canal=0, normalizar=True)
            X.append(extraer_features(audio))
            y.append(row["label"])
            ids.append(anon_id)
        except Exception as e:
            print(f"[ERROR] {anon_id}: {e}")

    return np.array(X), np.array(y), ids


def nombres_features(n_mfcc=N_MFCC):
    """Devuelve los nombres legibles de las 39 características."""
    return (
        [f"MFCC_{i+1} (media)" for i in range(n_mfcc)]
        + [f"MFCC_{i+1} (desv. est.)" for i in range(n_mfcc)]
        + [f"MFCC_{i+1} (delta)" for i in range(n_mfcc)]
    )