from fpdf import FPDF

class PDFGenerator:
    def generate_pdf(self, applicant_details, prediction, shap_summary, advice, output_path):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Loan Eligibility Report", ln=True, align='C')
        pdf.ln(10)

        pdf.cell(200, 10, txt=f"Applicant Details: {applicant_details}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
        pdf.cell(200, 10, txt=f"SHAP Summary: {shap_summary}", ln=True)
        pdf.cell(200, 10, txt=f"Advice: {', '.join(advice)}", ln=True)

        pdf.output(output_path)