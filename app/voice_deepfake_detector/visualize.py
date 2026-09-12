"""
Todas las visualizaciones del proyecto:
  1. Waveforms y espectrogramas (human vs synthetic)
  2. Boxplot de MFCCs
  3. PCA de MFCCs
  4. Importancia de características del modelo entrenado
"""
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.decomposition import PCA
from config import (
    MANIFEST_PATH, X_FEATURES_PATH, Y_LABELS_PATH,
    MODEL_PATH, PLOTS_DIR, SAMPLE_RATE,
)
from audio_utils import cargar_audio, nombres_features


def _guardar(nombre):
    ruta = os.path.join(PLOTS_DIR, nombre)
    plt.savefig(ruta, dpi=120, bbox_inches="tight")
    print(f"[OK] Guardado: {ruta}")


def plot_audio_crudo():
    """Compara waveform y espectrograma de un humano y un sintético."""
    df = pd.read_csv(MANIFEST_PATH)
    id_human = df[df["label"] == "human"].iloc[0]["anon_id"]
    id_synth = df[df["label"] == "synthetic"].iloc[0]["anon_id"]

    y_human = cargar_audio(id_human)
    y_synth = cargar_audio(id_synth)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for i, (y, id_, titulo, color) in enumerate([
        (y_human, id_human, "Human", "blue"),
        (y_synth, id_synth, "Synthetic", "red"),
    ]):
        axes[i, 0].set_title(f"{titulo}: {id_} - Waveform")
        librosa.display.waveshow(y, sr=SAMPLE_RATE, ax=axes[i, 0], color=color)

        axes[i, 1].set_title(f"{titulo}: {id_} - Spectrogram")
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(
            D, sr=SAMPLE_RATE, x_axis="time", y_axis="hz",
            ax=axes[i, 1], cmap="magma",
        )
    plt.tight_layout()
    _guardar("01_audio_crudo.png")
    plt.show()


def plot_boxplot_mfcc():
    """Boxplot de los MFCCs por clase."""
    X = np.load(X_FEATURES_PATH)
    y = np.load(Y_LABELS_PATH)

    # Solo los 13 primeros (medias) para el boxplot clásico
    cols = [f"MFCC_{i+1}" for i in range(13)]
    df = pd.DataFrame(X[:, :13], columns=cols)
    df["label"] = y
    df_melted = df.melt(id_vars="label", var_name="MFCC", value_name="Value")

    plt.figure(figsize=(15, 6))
    sns.boxplot(
        data=df_melted, x="MFCC", y="Value", hue="label",
        palette={"human": "skyblue", "synthetic": "salmon"},
    )
    plt.title("Distribución de los 13 Coeficientes MFCC por Clase")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    _guardar("02_boxplot_mfcc.png")
    plt.show()


def plot_pca():
    """PCA 2D de las features."""
    X = np.load(X_FEATURES_PATH)
    y = np.load(Y_LABELS_PATH)

    X_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(10, 6))
    for label, color in zip(["human", "synthetic"], ["skyblue", "salmon"]):
        mask = y == label
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=label, alpha=0.7, c=color, edgecolors="k",
        )
    plt.title("Visualización PCA (Human vs Synthetic)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _guardar("03_pca.png")
    plt.show()


def plot_importancia():
    """Importancia de las top 20 características del modelo."""
    modelo = joblib.load(MODEL_PATH)
    importancias = modelo.feature_importances_
    nombres = nombres_features()
    indices = np.argsort(importancias)[::-1][:20]

    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(importancias[indices] / max(importancias[indices]))
    plt.barh(range(len(indices)), importancias[indices], color=colors)
    plt.yticks(range(len(indices)), [nombres[i] for i in indices])
    plt.xlabel("Importancia relativa")
    plt.title("Top 20 Características Más Importantes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _guardar("04_importancia.png")
    plt.show()

    print("\nTop 10 características:")
    for i, idx in enumerate(indices[:10], 1):
        print(f"{i:2d}. {nombres[idx]:25s} -> {importancias[idx]:.4f}")


if __name__ == "__main__":
    plot_audio_crudo()
    plot_boxplot_mfcc()
    plot_pca()
    plot_importancia()