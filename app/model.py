import os
import joblib
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from app.audio_utils import extract_features

MODEL_PATH = "model.pkl"

def process_single_file(row, audio_dir):
    """Worker function to process a single audio file on a separate CPU core."""
    filepath = os.path.join(audio_dir, f"{row['anon_id']}.wav")
    if not os.path.exists(filepath):
        return None
        
    try:
        data, sr = sf.read(filepath)
        caller = data[:, 0]
        agent = data[:, 1]
        
        feats = extract_features(caller, agent, sr)
        label = 1 if row["label"] == "synthetic" else 0
        return feats, label, row["split"]
    except Exception:
        # Failsafe so one corrupted WAV file doesn't crash the entire batch
        return None

def train_and_save_model(manifest_path: str = "manifest.csv", audio_dir: str = "audio"):
    """Extracts features, caches them locally, and trains the classifier."""
    features_path = "features_cache.pkl"
    
    # 1. Load from cache if it exists to save 12 minutes on future runs
    if os.path.exists(features_path):
        print("Loading cached features from disk...")
        valid_results = joblib.load(features_path)
    else:
        manifest = pd.read_csv(manifest_path)
        total_files = len(manifest)
        print(f"Extracting features from {total_files} files using all CPU cores...")
        
        results = joblib.Parallel(n_jobs=-1, verbose=10)(
            joblib.delayed(process_single_file)(row, audio_dir) for _, row in manifest.iterrows()
        )
        valid_results = [r for r in results if r is not None]
        
        # Save the extracted data so we never have to compute this batch again
        joblib.dump(valid_results, features_path)
        print("Features cached successfully to features_cache.pkl")
    
    X = np.array([r[0] for r in valid_results])
    y = np.array([r[1] for r in valid_results])
    splits = np.array([r[2] for r in valid_results])
    
    train_mask = splits == "train"
    
    print("Training and calibrating classifier...")
    base_model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
    
    # 2. scikit-learn 1.4+ fix: Use internal CV instead of 'prefit'
    calibrated_model = CalibratedClassifierCV(estimator=base_model, cv=5, method='isotonic')
    calibrated_model.fit(X[train_mask], y[train_mask])
    
    joblib.dump(calibrated_model, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
    return calibrated_model

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_and_save_model()