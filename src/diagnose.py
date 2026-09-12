import pandas as pd
import numpy as np
import joblib
import os
from config import MANIFEST_PATH, MODEL_DIR
from features import extract_features

def main():
    df = pd.read_csv(MANIFEST_PATH)
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    # Detectar si el modelo fue entrenado con semántica
    use_semantic = any(
        fn.startswith("hesitation") or fn.startswith("caller_word")
        for fn in feature_names
    )
    print(f"Modelo espera {len(feature_names)} features (semantic={use_semantic})")

    print("\nExtrayendo features de todo el dataset (usa cache de Whisper)...")
    X, y = [], []
    for i, row in df.iterrows():
        feats = extract_features(row['anon_id'], include_semantic=use_semantic)
        X.append(feats)
        y.append(1 if row['label'] == 'synthetic' else 0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")

    X = pd.DataFrame(X).fillna(0)
    X = X.reindex(columns=feature_names, fill_value=0)
    y = np.array(y)

    # Acceder al clasificador interno si está calibrado
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
    else:
        base = model

    importances = pd.Series(base.feature_importances_, index=X.columns)
    print("\n=== TOP 25 FEATURES MÁS IMPORTANTES ===")
    print(importances.sort_values(ascending=False).head(25).to_string())

    print("\n=== ESTADÍSTICAS POR CLASE (top 15) ===")
    X = X.copy()
    X['label'] = y
    top_feats = importances.sort_values(ascending=False).head(15).index.tolist()
    stats = X.groupby('label')[top_feats].mean().T
    stats.columns = ['human', 'synthetic']
    stats['ratio'] = stats['synthetic'] / (stats['human'].abs() + 1e-9)
    print(stats.to_string())

if __name__ == "__main__":
    main()