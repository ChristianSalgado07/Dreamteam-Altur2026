"""
Baseline Human-vs-Synthetic classifiers for Channel 0, on the
complete-case modeling table (modeling/prepare_dataset.py). Simple
models, no hyperparameter tuning, no deep learning — the goal is a first
performance baseline to understand the data, not a final detector.

label: 1 = synthetic, 0 = human (source of truth: manifest.csv, via the
acoustic table). Channel 1 is NOT included as a training row — see
modeling/README.md.

Run with: python -m modeling.baseline
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modeling.prepare_dataset import MODELING_DATASET_COMPLETE_CSV, FEATURES_DIR, NON_FEATURE_COLS

PREDICTIONS_CSV = FEATURES_DIR / "modeling_predictions.csv"

RANDOM_STATE = 42


def load_split():
    df = pd.read_csv(MODELING_DATASET_COMPLETE_CSV)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = (df["label"] == "synthetic").astype(int)  # 1 = synthetic, 0 = human

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"

    return {
        "df": df, "feature_cols": feature_cols,
        "X_train": X[train_mask], "y_train": y[train_mask], "id_train": df.loc[train_mask, "anon_id"],
        "X_val": X[val_mask], "y_val": y[val_mask], "id_val": df.loc[val_mask, "anon_id"],
    }


def get_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def run():
    data = load_split()
    models = get_models()

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(data["X_train"], data["y_train"])
        fitted[name] = model

        for split_name, X, y, ids in (
            ("train", data["X_train"], data["y_train"], data["id_train"]),
            ("val", data["X_val"], data["y_val"], data["id_val"]),
        ):
            proba = model.predict_proba(X)[:, 1]
            pred = model.predict(X)
            for aid, true_label, p_label, p_proba in zip(ids, y, pred, proba):
                rows.append({
                    "anon_id": aid, "split": split_name, "model": name,
                    "true_label": int(true_label), "predicted_label": int(p_label),
                    "predicted_proba_synthetic": float(p_proba),
                })

    predictions = pd.DataFrame(rows)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Trained {len(models)} models on {len(data['X_train'])} train / {len(data['X_val'])} val recordings, "
          f"{len(data['feature_cols'])} features. Predictions -> {PREDICTIONS_CSV}")
    return fitted, data


if __name__ == "__main__":
    run()
