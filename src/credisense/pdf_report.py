from fpdf import FPDF
from typing import Dict, List


def generate_pdf(applicant: Dict, prediction: Dict, shap_summary: Dict, advice: List[str], out_path: str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "CrediSense - Loan Eligibility Report", ln=True)
    pdf.ln(4)

    pdf.cell(0, 8, "Applicant Details:", ln=True)
    for k, v in applicant.items():
        pdf.cell(0, 6, f"{k}: {v}", ln=True)

    pdf.ln(4)
    pdf.cell(0, 8, "Model Output:", ln=True)
    pdf.cell(0, 6, f"Decision: {prediction.get('label')}", ln=True)
    pdf.cell(0, 6, f"Probability: {prediction.get('probability'):.3f}", ln=True)

    pdf.ln(4)
    pdf.cell(0, 8, "Top SHAP Features:", ln=True)
    for name, val in shap_summary.get("top_features", [])[:10]:
        pdf.cell(0, 6, f"{name}: {val}", ln=True)

    pdf.ln(4)
    pdf.cell(0, 8, "Recommendations:", ln=True)
    for a in advice:
        pdf.multi_cell(0, 6, f"- {a}")

    pdf.output(out_path)
