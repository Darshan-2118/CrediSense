import sqlite3
import os
import json
from typing import Optional, Dict, Any

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "credisense.db"))


def _ensure_db_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def init_db(db_path: Optional[str] = None):
    db = db_path or DB_PATH
    _ensure_db_dir()
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS applicants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id INTEGER,
            label TEXT,
            probability REAL,
            shap_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS batch_tracker (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            count INTEGER DEFAULT 0
        )
        """
    )
    # ensure a single row in batch_tracker
    cur.execute("INSERT OR IGNORE INTO batch_tracker (id, count) VALUES (1, 0)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS retraining_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            num_records INTEGER,
            model_version TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def insert_applicant(payload: Dict[str, Any], db_path: Optional[str] = None) -> int:
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("INSERT INTO applicants (payload) VALUES (?)", (json.dumps(payload),))
    id_ = cur.lastrowid
    conn.commit()
    conn.close()
    return id_


def insert_prediction(applicant_id: int, label: str, prob: float, shap_summary: Dict, db_path: Optional[str] = None):
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (applicant_id, label, probability, shap_summary) VALUES (?, ?, ?, ?)",
        (applicant_id, label, float(prob), json.dumps(shap_summary)),
    )
    conn.commit()
    conn.close()


def add_training_record(payload: Dict[str, Any], db_path: Optional[str] = None):
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("INSERT INTO training_data (payload) VALUES (?)", (json.dumps(payload),))
    cur.execute("UPDATE batch_tracker SET count = count + 1 WHERE id = 1")
    conn.commit()
    conn.close()


def get_batch_count(db_path: Optional[str] = None) -> int:
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT count FROM batch_tracker WHERE id = 1")
    row = cur.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def reset_batch_count(db_path: Optional[str] = None):
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("UPDATE batch_tracker SET count = 0 WHERE id = 1")
    conn.commit()
    conn.close()


def log_retraining(num_records: int, model_version: str = "unknown", db_path: Optional[str] = None):
    db = db_path or DB_PATH
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("INSERT INTO retraining_logs (num_records, model_version) VALUES (?, ?)", (num_records, model_version))
    conn.commit()
    conn.close()
