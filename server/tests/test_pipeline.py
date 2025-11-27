import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from server.preprocessing.data_preprocessing import DataPreprocessor  # type: ignore
from server.ml_model.model import LoanEligibilityModel  # type: ignore
from server.explainability.shap_explainer import SHAPExplainer  # type: ignore
from server.advisory.advisory_engine import AdvisoryEngine  # type: ignore
from server.database.database import Database  # type: ignore
from server.pdf_generator.pdf_generator import PDFGenerator  # type: ignore

# NOTE: Editor linters may not resolve the `server` package if the workspace
# root is set to the `server` folder. Use the provided `run_tests.py` runner
# or set PYTHONPATH to the parent directory when running tests in the
# terminal/CI. The `# type: ignore` comments above silence missing-import
# diagnostics in the editor while keeping the imports functional at runtime.

class TestPipeline(unittest.TestCase):
    def test_preprocessing(self):
        preprocessor = DataPreprocessor()
        sample_data = {
            'income': [5000],
            'loan_amount': [200000],
            'debt_to_income_ratio': [0.4],
            'employment_type': ['Salaried'],
            'property_area': ['Urban']
        }
        processed_data = preprocessor.preprocess(sample_data)
        self.assertIsNotNone(processed_data)

    def test_model_prediction(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dummy_model.joblib'))
        model = LoanEligibilityModel(model_path)
        sample_data = [[5000, 200000, 0.4, 1, 0]]  # Example processed data
        prediction, probability = model.predict(sample_data)
        self.assertIsNotNone(prediction)
        self.assertIsNotNone(probability)

    def test_shap_explanation(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dummy_model.joblib'))
        model = LoanEligibilityModel(model_path)
        explainer = SHAPExplainer(model)
        sample_data = [[5000, 200000, 0.4, 1, 0]]
        shap_values, expected_value = explainer.explain(sample_data)
        self.assertIsNotNone(shap_values)
        self.assertIsNotNone(expected_value)

    def test_advisory_engine(self):
        advisory = AdvisoryEngine()
        input_data = {
            'cibil_score': 600,
            'debt_to_income_ratio': 0.5,
            'missed_emis': 2
        }
        prediction = 0.4
        advice = advisory.generate_advice(input_data, prediction)
        self.assertGreater(len(advice), 0)

    def test_database(self):
        db = Database(':memory:')
        db.insert_applicant('input_data', 'prediction', 0.8, 'shap_summary')
        applicants = db.fetch_all_applicants()
        self.assertEqual(len(applicants), 1)

    def test_pdf_generation(self):
        pdf_gen = PDFGenerator()
        pdf_gen.generate_pdf('details', 'prediction', 'shap_summary', ['advice1', 'advice2'], 'test_report.pdf')
        self.assertTrue(True)  # Check if no exceptions are raised

if __name__ == '__main__':
    unittest.main()