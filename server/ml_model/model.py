import joblib

class LoanEligibilityModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, data):
        return self.model.predict(data), self.model.predict_proba(data)[:, 1]