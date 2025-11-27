import os
import sys
import json

# Make sure src package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from credisense.advisory import generate_advice


def test_advice_low_cibil():
    applicant = {"cibil_score": 620, "debt_to_income": 0.2, "missed_emis": 0, "income": 40000, "loan_amount": 100000}
    adv = generate_advice(applicant)
    assert any("Improve your CIBIL" in a for a in adv)


def test_advice_high_dti_and_missed():
    applicant = {"cibil_score": 700, "debt_to_income": 0.6, "missed_emis": 2, "income": 30000, "loan_amount": 200000}
    adv = generate_advice(applicant)
    assert any("debt-to-income" in a for a in adv)
    assert any("missed EMIs" in a for a in adv)
