import os
import joblib
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from app.audio_utils import extract_features

MODEL_PATH = "model.pkl"

def train_and_save_model(manifest_path: str = "manifest.csv", audio_dir: str = "audio"):
    """Extracts features from the dataset and trains a calibrated classifier."""
    manifest = pd.read_csv(manifest_path)
    
    X, y, splits = [], [], []
    print("Extracting features from audio files...")
    
    for _, row in manifest.iterrows():
        filepath = os.path.join(audio_dir, f"{row['anon_id']}.wav")
        if not os.path.exists(filepath):
            continue
        
        data, sr = sf.read(filepath)
        caller = data[:, 0]
        agent = data[:, 1]
        
        feats = extract_features(caller, agent, sr)
        X.append(feats)
        y.append(1 if row["label"] == "synthetic" else 0)
        splits.append(row["split"])
    
    X = np.array(X)
    y = np.array(y)
    splits = np.array(splits)
    
    train_mask = splits == "train"
    val_mask = splits == "val"
    
    base_model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
    base_model.fit(X[train_mask], y[train_mask])
    
    calibrated_model = CalibratedClassifierCV(estimator=base_model, cv='prefit', method='isotonic')
    calibrated_model.fit(X[val_mask], y[val_mask])
    
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
    return calibrated_model

def load_or_train_model():
    """Loads existing trained model or triggers training if not found."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    print("Model not found. Initializing training...")
    return train_and_save_model()