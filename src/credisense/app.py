from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .preprocessing import preprocess
from .ml_model import load_model, predict
from .explainability import explain_model
from .advisory import generate_advice
from .database import init_db, insert_applicant, insert_prediction, add_training_record
from .training import retrain_if_needed
import os

app = FastAPI(title="CrediSense Backend")


class Applicant(BaseModel):
    income: float = 0
    loan_amount: float = 0
    cibil_score: float = 0
    previous_loans: int = 0
    missed_emis: int = 0
    employment_type: str = "salaried"
    debt_to_income: float = 0
    age: int = 30
    dependents: int = 0


@app.on_event("startup")
def startup():
    # Ensure DB exists
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(applicant: Applicant):
    app_dict = applicant.dict()
    # store applicant
    aid = insert_applicant(app_dict)

    X = preprocess(app_dict)
    model = load_model()
    label, proba = predict(model, X.values[0])
    shap_summary = explain_model(model, X, feature_names=X.columns.tolist())
    advice = generate_advice(app_dict)

    insert_prediction(aid, label, proba, shap_summary)

    return {"label": label, "probability": proba, "shap": shap_summary, "advice": advice}


@app.post("/training/add")
def add_training(applicant: dict):
    # Add a training record (raw payload) and trigger retraining if threshold reached
    add_training_record(applicant)
    retrained = retrain_if_needed()
    return {"accepted": True, "retrained": retrained}


@app.post("/retrain/force")
def retrain_force():
    # Force retrain regardless of batch count
    did = retrain_if_needed(batch_threshold=0)
    return {"retrained": did}
