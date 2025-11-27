import sqlite3
import json
import pandas as pd


class Database:
    def __init__(self, db_path):
        # Enable row factory for convenience
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS applicants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_data TEXT,
                    prediction TEXT,
                    probability REAL,
                    shap_summary TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    features_json TEXT,
                    label INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS retraining_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    records_used INTEGER,
                    model_path TEXT
                )
            ''')

            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS batch_tracker (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    count INTEGER DEFAULT 0
                )
            ''')

            # Ensure a single row exists in batch_tracker
            cur = self.connection.execute('SELECT count(*) as c FROM batch_tracker')
            if cur.fetchone()['c'] == 0:
                self.connection.execute('INSERT INTO batch_tracker (id, count) VALUES (1, 0)')

    # Applicants
    def insert_applicant(self, input_data, prediction, probability, shap_summary):
        with self.connection:
            self.connection.execute(
                'INSERT INTO applicants (input_data, prediction, probability, shap_summary) VALUES (?, ?, ?, ?)',
                (json.dumps(input_data), json.dumps(prediction), probability, json.dumps(shap_summary))
            )

    def fetch_all_applicants(self):
        with self.connection:
            rows = self.connection.execute('SELECT * FROM applicants').fetchall()
            return [dict(r) for r in rows]

    # Training data interface
    def insert_training_record(self, features: dict, label: int):
        with self.connection:
            self.connection.execute(
                'INSERT INTO training_data (features_json, label) VALUES (?, ?)',
                (json.dumps(features), int(label))
            )
            # increment batch tracker
            self.increment_batch_count(1)

    def fetch_training_dataframe(self) -> pd.DataFrame:
        with self.connection:
            rows = self.connection.execute('SELECT features_json, label FROM training_data').fetchall()
            records = [json.loads(r['features_json']) for r in rows]
            labels = [r['label'] for r in rows]
            if len(records) == 0:
                return pd.DataFrame()
            df = pd.DataFrame(records)
            df['loan_approved'] = labels
            return df

    # Batch tracker
    def get_batch_count(self) -> int:
        with self.connection:
            row = self.connection.execute('SELECT count FROM batch_tracker WHERE id = 1').fetchone()
            return int(row['count'])

    def increment_batch_count(self, n: int = 1):
        with self.connection:
            self.connection.execute('UPDATE batch_tracker SET count = count + ? WHERE id = 1', (n,))

    def reset_batch_count(self):
        with self.connection:
            self.connection.execute('UPDATE batch_tracker SET count = 0 WHERE id = 1')

    # Retraining logs
    def log_retraining(self, records_used: int, model_path: str):
        with self.connection:
            self.connection.execute(
                'INSERT INTO retraining_logs (records_used, model_path) VALUES (?, ?)',
                (records_used, model_path)
            )

    def close(self):
        try:
            self.connection.close()
        except Exception:
            pass