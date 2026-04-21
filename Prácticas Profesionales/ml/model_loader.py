from pathlib import Path
import joblib


def load_model_if_exists(model_path: Path):
    if not model_path.exists():
        return None

    try:
        artifact = joblib.load(model_path)
        return artifact
    except Exception:
        return None