import sys
import os

# Ensure project root is on sys.path so 'server' package resolves when running script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from server.database.database import Database
from server.training.model_training import ModelTrainer

DB_PATH = 'app_data.db'
MODEL_PATH = 'model.joblib'

# create db and insert sample training records
db = Database(DB_PATH)

sample_records = [
    ({'income':5000, 'loan_amount':200000, 'debt_to_income_ratio':0.4, 'employment_type':'Salaried', 'property_area':'Urban'}, 1),
    ({'income':8000, 'loan_amount':100000, 'debt_to_income_ratio':0.2, 'employment_type':'Self-Employed', 'property_area':'Rural'}, 0),
    ({'income':6000, 'loan_amount':150000, 'debt_to_income_ratio':0.3, 'employment_type':'Salaried', 'property_area':'Urban'}, 1),
]

for feat, label in sample_records:
    db.insert_training_record(feat, label)

trainer = ModelTrainer(DB_PATH, MODEL_PATH)
info = trainer.train_from_db()
print('Retraining info:', info)
trainer.close()
