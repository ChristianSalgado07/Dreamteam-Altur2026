import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import MANIFEST_PATH
from features import extract_features

def main():
    df = pd.read_csv(MANIFEST_PATH)

    print("Extrayendo features de todas las llamadas...")
    X, y = [], []
    for i, row in df.iterrows():
        feats = extract_features(row['anon_id'])
        X.append(feats)
        y.append(1 if row['label'] == 'synthetic' else 0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(df)}")

    X = pd.DataFrame(X).fillna(0)
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== ROC AUC ===")
    aucs = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    print("Por fold:", aucs)
    print(f"Media: {aucs.mean():.4f} ± {aucs.std():.4f}")

    print("\n=== ACCURACY ===")
    accs = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print("Por fold:", accs)
    print(f"Media: {accs.mean():.4f} ± {accs.std():.4f}")

if __name__ == "__main__":
    main()  