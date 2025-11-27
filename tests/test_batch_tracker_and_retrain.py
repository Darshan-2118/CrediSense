import os
import sys
import tempfile
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from credisense import database, training


def test_retrain_trigger(tmp_path):
    # Use a temp DB and temp model dir to avoid touching workspace
    db_file = str(tmp_path / "test_credisense.db")
    database.init_db(db_file)

    # prepare training payloads
    for i in range(30):
        payload = {"cibil_score": 600 + (i % 50), "income": 30000 + i * 100}
        database.add_training_record(payload, db_path=db_file)

    # patch training model dir to tmp
    tmp_model_dir = str(tmp_path / "models")
    training.MODEL_DIR = tmp_model_dir
    training.MODEL_PATH = os.path.join(tmp_model_dir, "model.joblib")

    did = training.retrain_if_needed(db_path=db_file, batch_threshold=30)
    assert did is True
    assert os.path.exists(training.MODEL_PATH)
