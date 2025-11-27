from typing import List, Dict


def generate_advice(applicant: Dict) -> List[str]:
    """Return list of advice strings based on simple rule engine.

    Rules are intentionally simple and transparent.
    """
    adv = []
    cibil = float(applicant.get("cibil_score", 0) or 0)
    dti = float(applicant.get("debt_to_income", 0) or 0)
    missed = int(applicant.get("missed_emis", 0) or 0)
    loan_amount = float(applicant.get("loan_amount", 0) or 0)
    income = float(applicant.get("income", 0) or 0)

    if cibil < 650:
        adv.append("Improve your CIBIL score: pay bills on time and reduce credit utilization.")
    elif cibil < 700:
        adv.append("Increase your CIBIL score to 700+ for better offers.")

    if dti > 0.4:
        adv.append("Reduce your debt-to-income ratio by lowering debt or increasing income.")

    if missed > 0:
        adv.append("Clear missed EMIs and maintain consistent repayments to improve risk profile.")

    # EMI burden heuristic
    if income > 0 and loan_amount / (income + 1) > 0.5:
        adv.append("Consider reducing requested loan amount to lower EMI burden.")

    # Suggest co-applicant
    if income < 20000 and loan_amount > 500000:
        adv.append("Adding a co-applicant with stable income could improve approval odds.")

    if not adv:
        adv.append("No major issues detected — maintain current profile and repayment discipline.")

    return adv
