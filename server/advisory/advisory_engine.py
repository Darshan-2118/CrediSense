class AdvisoryEngine:
    def generate_advice(self, input_data, prediction):
        advice = []

        if input_data['cibil_score'] < 650:
            advice.append("Improve your CIBIL score above 650.")
        if input_data['debt_to_income_ratio'] > 0.4:
            advice.append("Reduce your debt-to-income ratio below 40%.")
        if input_data['missed_emis'] > 0:
            advice.append("Clear your missed EMIs to improve your creditworthiness.")
        if prediction < 0.5:
            advice.append("Consider adding a co-applicant to strengthen your application.")

        return advice