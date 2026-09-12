import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
SAMPLE_RATE = 8000