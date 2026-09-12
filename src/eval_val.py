import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from config import MANIFEST_PATH, MODEL_DIR
from features import extract_features
from predict import Detector

def main():
    df = pd.read_csv(MANIFEST_PATH)
    val_df = df[df['split'] == 'val']

    print(f"Evaluando en val: {len(val_df)} llamadas")

    det = Detector()

    results = []
    for _, row in val_df.iterrows():
        feats = extract_features(row['anon_id'])
        X = pd.DataFrame([feats])
        if det.feature_names is not None:
            X = X.reindex(columns=det.feature_names, fill_value=0)
        X = X.fillna(0)

        raw = float(det.model.predict_proba(X)[0, 1])
        from predict import _soften, CONF_MIN, CONF_MAX
        soft = max(CONF_MIN, min(CONF_MAX, _soften(raw)))

        results.append({
            'id': row['anon_id'],
            'label': row['label'],
            'raw': raw,
            'soft': soft,
            'pred_raw': 'synthetic' if raw > 0.5 else 'human',
            'pred_soft': 'synthetic' if soft > 0.5 else 'human',
        })

    r = pd.DataFrame(results)
    y = (r['label'] == 'synthetic').astype(int)

    print(f"\n=== ANTES (probs crudas) ===")
    print(f"Rango: [{r['raw'].min():.4f}, {r['raw'].max():.4f}]")
    print(f"ROC AUC: {roc_auc_score(y, r['raw']):.4f}")
    print(f"Brier: {brier_score_loss(y, r['raw']):.4f}")

    print(f"\n=== DESPUÉS (con temperature T=3) ===")
    print(f"Rango: [{r['soft'].min():.4f}, {r['soft'].max():.4f}]")
    print(f"ROC AUC: {roc_auc_score(y, r['soft']):.4f}")
    print(f"Brier: {brier_score_loss(y, r['soft']):.4f}")

    print(f"\n=== DISTRIBUCIÓN SUAVIZADA ===")
    print("Sintéticos:")
    print(r[r['label'] == 'synthetic']['soft'].describe())
    print("\nHumanos:")
    print(r[r['label'] == 'human']['soft'].describe())

    print(f"\n=== MUESTRA INDIVIDUAL ===")
    for _, row in r.iterrows():
        print(f"{row['id']} | real={row['label']:<10} | raw={row['raw']:.4f} | soft={row['soft']:.4f}")

if __name__ == "__main__":
    main()
