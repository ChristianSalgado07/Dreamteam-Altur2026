"""
Cache en disco de features semanticas por anon_id.
Evita re-transcribir si el entrenamiento se reinicia.
"""
import os
import pickle

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "semantic_cache"
)


def _ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cached(anon_id):
    _ensure_dir()
    path = os.path.join(CACHE_DIR, f"{anon_id}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def set_cached(anon_id, semantic_feats):
    _ensure_dir()
    path = os.path.join(CACHE_DIR, f"{anon_id}.pkl")
    with open(path, "wb") as f:
        pickle.dump(semantic_feats, f)


def cache_stats():
    _ensure_dir()
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    return len(files)
