import shap
import numpy as np

class SHAPExplainer:
    def __init__(self, model):
        # Accept either a raw model or a wrapper that holds the model
        raw_model = getattr(model, 'model', model)
        self.explainer = shap.TreeExplainer(raw_model)

    def explain(self, data):
        # Ensure data is numpy array or DataFrame
        if isinstance(data, list):
            data = np.array(data)
        shap_values = self.explainer.shap_values(data)
        # expected_value may be a list for multiclass; return as-is
        return shap_values, getattr(self.explainer, 'expected_value', None)