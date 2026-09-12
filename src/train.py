import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
import joblib
import os
from config import MANIFEST_PATH, MODEL_DIR
from features import extract_features, extract_features_with_augmentation

INCLUDE_SEMANTIC = True
USE_AUGMENTATION = True
N_AUGMENTS = 1  # 1 augment por llamada → dataset x2

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=4,
    min_samples_split=8,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
)


def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Total llamadas: {len(df)}")
    print(df['label'].value_counts())

    print(f"\nExtrayendo features (semantic={INCLUDE_SEMANTIC}, augment={USE_AUGMENTATION})...")
    X, y = [], []
    for i, row in df.iterrows():
        label = 1 if row['label'] == 'synthetic' else 0
        if USE_AUGMENTATION:
            feats_list = extract_features_with_augmentation(
                row['anon_id'], n_augments=N_AUGMENTS, include_semantic=INCLUDE_SEMANTIC
            )
        else:
            feats_list = [extract_features(row['anon_id'], include_semantic=INCLUDE_SEMANTIC)]

        for f in feats_list:
            X.append(f)
            y.append(label)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)} llamadas procesadas ({len(X)} muestras)")

    X = pd.DataFrame(X).fillna(0)
    y = np.array(y)
    print(f"\nDataset final: {X.shape[0]} muestras, {X.shape[1]} features")

    print("\n=== 5-FOLD CROSS-VALIDATION ===")
    base_model = RandomForestClassifier(**RF_PARAMS)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(base_model, X, y, cv=skf, scoring='roc_auc')
    accs = cross_val_score(base_model, X, y, cv=skf, scoring='accuracy')
    print(f"ROC AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")

    print("\n=== ENTRENANDO MODELO FINAL ===")
    base_model.fit(X, y)
    calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated.fit(X, y)

    probs = calibrated.predict_proba(X)[:, 1]
    print(f"Brier score: {brier_score_loss(y, probs):.4f}")
    print(f"ROC AUC: {roc_auc_score(y, probs):.4f}")
    print(f"Rango de probs: [{probs.min():.4f}, {probs.max():.4f}]")

    y_pred = (probs > 0.5).astype(int)
    print("\n=== REPORTE FINAL ===")
    print(classification_report(y, y_pred, target_names=['human', 'synthetic']))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(calibrated, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))
    print(f"\nModelo guardado.")


if __name__ == "__main__":
    main()