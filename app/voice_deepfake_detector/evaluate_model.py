"""
Validación cruzada del modelo con Stratified K-Fold.
Reporta accuracy, precision, recall y F1.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from config import X_FEATURES_PATH, Y_LABELS_PATH, CV_FOLDS, RANDOM_STATE
from train_model import crear_modelo, binarizar

# Carga de datos
def main():
    X = np.load(X_FEATURES_PATH)
    y = binarizar(np.load(Y_LABELS_PATH))
    
    # Config val cruzada
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    metricas = ["accuracy", "precision", "recall", "f1"]
    resultados = cross_validate(crear_modelo(), X, y, cv=cv, scoring=metricas)

    print(f"\n Validación Cruzada ({CV_FOLDS}-Fold) ")
    for m in metricas:
        s = resultados[f"test_{m}"]
        print(f"\n{m.upper()}:")
        print(f"  Por fold: {[f'{v:.4f}' for v in s]}")
        print(f"  Media:    {s.mean():.4f}")
        print(f"  Desv.:    {s.std():.4f}")
        print(f"  Rango:    [{s.min():.4f}, {s.max():.4f}]")


if __name__ == "__main__":
    main()