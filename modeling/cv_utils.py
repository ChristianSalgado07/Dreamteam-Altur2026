"""
Shared, methodologically-careful evaluation utilities for the modeling
diagnostics (modeling/diagnostics.py). Used so every experiment in that
module (feature-family ablation, controlled elimination, suspicious
simple baselines, original-vs-denoised, ...) is scored the same way.

Scaling rule enforced everywhere here: StandardScaler is placed INSIDE
each model's sklearn Pipeline, so:
  - in plain train/val scoring, the scaler is fit on X_train only
    (Pipeline.fit(X_train, y_train)) and applied to X_val via that same
    fitted scaler (Pipeline.predict on X_val calls .transform, not
    .fit_transform) — validation data never influences the scaler.
  - in cross-validation, sklearn's cross_validate() clones and re-fits
    the WHOLE pipeline (scaler included) on each fold's training portion
    and only transforms the held-out fold with that fold's fitted
    scaler — never fit on the complete dataset before CV.
Random Forest is never wrapped in a scaler (tree splits are scale-invariant).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
N_CV_FOLDS = 5

# one fixed splitter, reused for every feature-set comparison in this
# module so the ablation experiments are compared on the SAME folds
CV_SPLITTER = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
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


def uses_standardization(model_name: str) -> bool:
    return model_name in ("logistic_regression", "svm")


def cross_validate_model(model, X, y) -> dict:
    """Mean +/- std over N_CV_FOLDS stratified folds of the TRAIN data
    only. Scaling (where used) is fit fresh inside each fold by
    cross_validate() itself — see module docstring."""
    result = cross_validate(model, X, y, cv=CV_SPLITTER, scoring=CV_SCORING, n_jobs=1)
    out = {}
    for metric in CV_SCORING:
        scores = result[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = float(np.mean(scores))
        out[f"cv_{metric}_std"] = float(np.std(scores))
    return out


def val_metrics(model, X_train, y_train, X_val, y_val) -> dict:
    """Fits fresh on X_train only, scores on X_val. If `model` is a
    Pipeline with a scaler step, the scaler is fit on X_train inside
    .fit() and only .transform()-ed for X_val — never re-fit on val."""
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "val_accuracy": accuracy_score(y_val, pred),
        "val_precision": precision_score(y_val, pred, zero_division=0),
        "val_recall": recall_score(y_val, pred, zero_division=0),
        "val_f1": f1_score(y_val, pred, zero_division=0),
        "val_roc_auc": roc_auc_score(y_val, proba) if len(set(y_val)) > 1 else float("nan"),
    }


def evaluate_feature_set(X_train, y_train, X_val, y_val, feature_cols, label: str) -> pd.DataFrame:
    """Runs every model in get_models() on one feature subset, both
    plain train/val and stratified CV-on-train. Returns one row per
    model."""
    rows = []
    for name, model in get_models().items():
        row = {
            "experiment": label,
            "model": name,
            "n_features": len(feature_cols),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "standardized": uses_standardization(name),
        }
        row.update(val_metrics(model, X_train[feature_cols], y_train, X_val[feature_cols], y_val))
        # re-instantiate for CV (val_metrics already fit this instance on train)
        cv_model = get_models()[name]
        row.update(cross_validate_model(cv_model, X_train[feature_cols], y_train))
        rows.append(row)
    return pd.DataFrame(rows)
