import random
import os
from server.database.database import Database
from server.training.model_training import ModelTrainer

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'server.db'))

# Create some synthetic training records
def make_record(i):
    return {
        'income': random.randint(2000, 15000),
        'cibil_score': random.randint(300, 900),
        'loan_amount': random.randint(50000, 500000),
        'previous_loans': random.randint(0, 5),
        'missed_emis': random.randint(0, 3),
        'employment_type': random.choice(['Salaried', 'Self-Employed', 'Unemployed']),
        'debt_to_income_ratio': round(random.uniform(0.05, 0.6), 2),
        'age': random.randint(21, 65),
        'dependents': random.randint(0, 4),
        'loan_approved': random.choice([0, 1])
    }


def populate(n=30):
    db = Database(DB_PATH)
    for i in range(n):
        rec = make_record(i)
        db.insert_training_record(rec, rec['loan_approved'])
    print(f'Inserted {n} synthetic training records into {DB_PATH}')
    db.close()


if __name__ == '__main__':
    populate(30)
    trainer = ModelTrainer(DB_PATH, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'model.joblib')))
    info = trainer.retrain_if_needed(batch_threshold=30)
    if info is None:
        print('No retraining performed; forcing retrain')
        info = trainer.train_from_db()
    print('Retrain info:', info)
    trainer.close()
