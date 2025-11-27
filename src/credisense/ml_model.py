import os
import joblib
import numpy as np
from typing import Tuple

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, "model.joblib"))


def _ensure_model_dir():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def load_model():
    """Load a serialized model. If absent, return None.
    The training pipeline should create a model at `MODEL_PATH`.
    """
    _ensure_model_dir()
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict(model, X) -> Tuple[str, float]:
    """Return label ('Eligible' or 'Not Eligible') and probability for applicant X.

    X may be a 2D array-like or DataFrame row.
    """
    if model is None:
        # fallback dummy behaviour
        # No model available — return Not Eligible with zero confidence
        return "Not Eligible", 0.0

    # Ensure numpy array
    xp = np.asarray(X)
    if xp.ndim == 1:
        xp = xp.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(xp)[0, 1])
    else:
        # fallback: use decision_function or predict
        try:
            proba = float(model.decision_function(xp)[0])
            # squash to 0-1
            proba = 1 / (1 + np.exp(-proba))
        except Exception:
            proba = float(model.predict(xp)[0]) if hasattr(model, "predict") else 0.0

    label = "Eligible" if proba >= 0.5 else "Not Eligible"
    return label, proba
