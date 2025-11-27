import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .database import get_batch_count, reset_batch_count, init_db, DB_PATH, log_retraining
from .preprocessing import preprocess

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")


def _ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)


def retrain_if_needed(db_path: str = None, batch_threshold: int = 30) -> bool:
    """Check batch count and retrain model when threshold is reached.

    Returns True if retraining occurred.
    """
    dbp = db_path or DB_PATH
    init_db(dbp)
    count = get_batch_count(dbp)
    if count < batch_threshold:
        return False

    # Load training data rows
    import sqlite3

    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute("SELECT payload FROM training_data")
    rows = cur.fetchall()
    conn.close()

    # Convert JSON payloads into DataFrame
    import json

    payloads = [json.loads(r[0]) for r in rows]
    if not payloads:
        reset_batch_count(dbp)
        return False

    df = pd.DataFrame(payloads)
    # Preprocess each row via preprocess (which returns a DataFrame)
    Xs = []
    for row in df.to_dict(orient="records"):
        Xs.append(preprocess(row).iloc[0].values)

    X = pd.DataFrame(Xs)

    # Fake label if not present: here we'll try to use 'label' column if present
    if "label" in df.columns:
        y = df["label"].astype(int)
    else:
        # No labels: create proxy using cibil_score > 650
        y = (df.get("cibil_score", pd.Series([0]*len(df))) > 650).astype(int)

    # Train a simple pipeline
    _ensure_dirs()
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=50, random_state=42))])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, MODEL_PATH)

    # Reset batch counter and log
    reset_batch_count(dbp)
    log_retraining(len(payloads), model_version="v1", db_path=dbp)
    return True
