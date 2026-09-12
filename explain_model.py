import joblib
import numpy as np
from sklearn.inspection import permutation_importance

def extract_weights():
    print("Loading cache and model...")
    features_list = joblib.load("features_cache.pkl")
    X = np.array([r[0] for r in features_list])
    y = np.array([r[1] for r in features_list])
    
    model = joblib.load("model.pkl")

    print("Calculating feature importance (this takes a few seconds)...")
    # n_jobs=-1 uses all CPU cores to run the permutations instantly
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    # Normalize the raw mathematical importances into percentages
    importances = result.importances_mean
    importances = 100.0 * (importances / importances.sum())

    # Map the 37 array indices back to your hybrid architecture
    timing_env = np.sum(importances[0:8])
    caller_mfcc = np.sum(importances[8:21])
    agent_mfcc = np.sum(importances[21:34])
    semantics = np.sum(importances[34:37])

    print("\n--- EXACT MODEL DECISION WEIGHTS ---")
    print(f"Timing & Environment (Overlaps, Latency, Noise): {timing_env:.1f}%")
    print(f"Caller Acoustics (Timbre, Spectral Shape): {caller_mfcc:.1f}%")
    print(f"Agent Acoustics (Baseline Context): {agent_mfcc:.1f}%")
    print(f"Semantics (AI Tells, Hallucinations, Fillers): {semantics:.1f}%\n")

if __name__ == "__main__":
    extract_weights()