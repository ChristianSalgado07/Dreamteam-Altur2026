import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from config import MANIFEST_PATH, MODEL_DIR
from features import extract_features

def main():
    df = pd.read_csv(MANIFEST_PATH)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']

    print("Extrayendo features de train...")
    X_train, y_train = [], []
    for _, row in train_df.iterrows():
        feats = extract_features(row['anon_id'])
        X_train.append(feats)
        y_train.append(1 if row['label'] == 'synthetic' else 0)

    print("Extrayendo features de val...")
    X_val, y_val = [], []
    for _, row in val_df.iterrows():
        feats = extract_features(row['anon_id'])
        X_val.append(feats)
        y_val.append(1 if row['label'] == 'synthetic' else 0)

    X_train = pd.DataFrame(X_train).fillna(0)
    X_val = pd.DataFrame(X_val).fillna(0)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, y_pred))
    print("ROC AUC:", roc_auc_score(y_val, y_prob))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    print("Modelo guardado en models/model.pkl")

if __name__ == "__main__":
    main()