import soundfile as sf
import numpy as np
from pathlib import Path
from app.audio_utils import extract_features

def compare_calls(fail_id, pass_id):
    # Dynamically search the entire project for the exact WAV files
    fail_path = next(Path('.').rglob(f"{fail_id}.wav"), None)
    pass_path = next(Path('.').rglob(f"{pass_id}.wav"), None)

    if not fail_path or not pass_path:
        print(f"File not found!\nLooking for: {fail_id}.wav -> Found: {fail_path}")
        print(f"Looking for: {pass_id}.wav -> Found: {pass_path}")
        return

    # Load the dynamically found files
    fail_data, sr_fail = sf.read(fail_path)
    pass_data, sr_pass = sf.read(pass_path)

    # Slice both to the exact 15-30 second window used in your endpoint
    fail_slice = fail_data[sr_fail*15 : sr_fail*30]
    pass_slice = pass_data[sr_pass*15 : sr_pass*30]

    # Extract the 40-feature array for both
    fail_features = extract_features(fail_slice[:, 0], fail_slice[:, 1], sr_fail)
    pass_features = extract_features(pass_slice[:, 0], pass_slice[:, 1], sr_pass)

    print(f"{'Feature Index':<15} | {'Call 14 (Scream)':<20} | {'Call 7 (Clean)':<20} | {'Difference'}")
    print("-" * 75)
    
    for i, (f_val, p_val) in enumerate(zip(fail_features, pass_features)):
        diff = abs(f_val - p_val)
        # Flag massive deviations to easily spot the anomalies
        marker = "<--- ANOMALY" if diff > (abs(p_val) * 1.5) else "" 
        print(f"Feature {i:<8} | {f_val:<20.4f} | {p_val:<20.4f} | {diff:.4f} {marker}")

# Execute using just the unique file IDs
compare_calls("call_ca184bb77a63", "call_be72006df132")