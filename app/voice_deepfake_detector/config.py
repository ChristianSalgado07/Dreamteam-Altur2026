"""
Configuración global del proyecto.
Todas las rutas y constantes viven aquí.
"""
import os

# Rutas base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "altur-challenge-audio", "audio")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Archivos de salida
X_FEATURES_PATH = os.path.join(OUTPUT_DIR, "X_enhanced.npy")
Y_LABELS_PATH = os.path.join(OUTPUT_DIR, "y_labels.npy")
IDS_PATH = os.path.join(OUTPUT_DIR, "ids_labels.csv")
MODEL_PATH = os.environ.get(
    "MODEL_PATH_OVERRIDE",
    os.path.join(OUTPUT_DIR, "modelo_final_audio.pkl"),
)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Parámetros de audio
SAMPLE_RATE = 16000
N_MFCC = 13

# Parámetros del modelo
# TEST_SIZE: dejado como referencia, no se usa en la versión final
# (migramos a validación cruzada + splits speaker-disjoint del manifest)
# TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Crear carpetas si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)