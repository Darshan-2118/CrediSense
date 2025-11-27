import os
import joblib
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from server.database.database import Database
from server.preprocessing.data_preprocessing import DataPreprocessor


class ModelTrainer:
    def __init__(self, db_path: str, model_path: str):
        self.db_path = db_path
        self.model_path = model_path
        self.db = Database(db_path)
        self.preprocessor = DataPreprocessor()

    def train_from_db(self):
        """Load training data from SQLite, preprocess, train model and save it.

        Returns: dict with training info (records_used, model_path)
        """
        df = self.db.fetch_training_dataframe()
        if df.empty:
            raise ValueError("No training data available in the database")

        # Separate features and label
        y = df['loan_approved'].astype(int)
        X = df.drop(columns=['loan_approved'])

        # Preprocess: fit the preprocessor on training data
        X_preprocessed = self.preprocessor.preprocess(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

        # Train model (RandomForest for demo)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Ensure target directory exists
        os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
        joblib.dump(model, self.model_path)

        # Log retraining
        records_used = len(df)
        self.db.log_retraining(records_used, self.model_path)
        # Reset batch counter after successful retrain
        self.db.reset_batch_count()

        return {"records_used": records_used, "model_path": self.model_path}

    def retrain_if_needed(self, batch_threshold: int = 30):
        count = self.db.get_batch_count()
        if count >= batch_threshold:
            return self.train_from_db()
        return None

    def close(self):
        self.db.close()


def main():
    parser = argparse.ArgumentParser(description='Training runner: train model from SQLite training_data')
    parser.add_argument('--db', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'server.db')),
                        help='Path to SQLite DB')
    parser.add_argument('--model', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'model.joblib')),
                        help='Path to save trained model')
    parser.add_argument('--force', action='store_true', help='Force retraining regardless of batch count')
    parser.add_argument('--threshold', type=int, default=30, help='Batch size threshold to trigger retraining')

    args = parser.parse_args()
    trainer = ModelTrainer(args.db, args.model)
    try:
        if args.force:
            info = trainer.train_from_db()
            print('Forced retrain complete:', info)
        else:
            result = trainer.retrain_if_needed(batch_threshold=args.threshold)
            if result is None:
                print(f'Batch count below threshold ({args.threshold}); no retraining performed.')
            else:
                print('Retraining performed:', result)
    finally:
        trainer.close()


if __name__ == '__main__':
    main()