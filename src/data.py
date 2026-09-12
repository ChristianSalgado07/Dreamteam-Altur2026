import pandas as pd
import soundfile as sf
import numpy as np
import os
from config import MANIFEST_PATH, RAW_DIR

def load_manifest():
    return pd.read_csv(MANIFEST_PATH)

def load_audio(anon_id):
    path = os.path.join(RAW_DIR, f"{anon_id}.wav")
    y, sr = sf.read(path, always_2d=True)  # shape (samples, channels)
    caller = y[:, 0]
    agent = y[:, 1] if y.shape[1] > 1 else np.zeros_like(caller)
    return caller, agent, sr