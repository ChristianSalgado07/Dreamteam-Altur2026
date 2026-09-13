"""
Entrena el modelo XGBoost final con el 100% de los datos.
Si no existen las features, las extrae primero.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from config import (
    MANIFEST_PATH, X_FEATURES_PATH, Y_LABELS_PATH, IDS_PATH,
    MODEL_PATH, RANDOM_STATE,
)
from audio_utils import extraer_features_dataset

"""Carga las features si existen; si no, las extrae."""
def get_or_create_features():
    if os.path.exists(X_FEATURES_PATH) and os.path.exists(Y_LABELS_PATH):
        print("[Info] Cargando features existentes...")
        X = np.load(X_FEATURES_PATH)
        y = np.load(Y_LABELS_PATH)
    else:
        print("[Info] Extrayendo features por primera vez...")
        X, y, ids = extraer_features_dataset(MANIFEST_PATH)
        np.save(X_FEATURES_PATH, X)
        np.save(Y_LABELS_PATH, y)
        pd.DataFrame({"anon_id": ids, "label": y}).to_csv(IDS_PATH, index=False)
    print(f"[Info] X={X.shape}, y={y.shape}")
    return X, y

"""Convierte 'human'/'synthetic' a 0/1."""
def binarizar(y):
    return np.where(y == "human", 0, 1)

"""XGBoost con los hiperparámetros validados."""
def crear_modelo():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )


def main():
    X, y = get_or_create_features()
    y_bin = binarizar(y)

    modelo = crear_modelo()
    modelo.fit(X, y_bin)
    joblib.dump(modelo, MODEL_PATH)
    print(f"[Ok] Modelo final guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()