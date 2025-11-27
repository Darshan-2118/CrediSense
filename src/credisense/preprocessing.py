import pandas as pd


def preprocess(applicant: dict) -> pd.DataFrame:
    """Convert applicant dict to processed DataFrame suitable for model.

    This is a lightweight preprocessing pipeline: fills missing values,
    maps employment types, and ensures numeric columns exist.
    """
    df = pd.DataFrame([applicant])

    # Basic columns we expect — add missing ones with NaN
    expected_cols = [
        "income",
        "loan_amount",
        "cibil_score",
        "previous_loans",
        "missed_emis",
        "employment_type",
        "debt_to_income",
        "age",
        "dependents",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Fill numeric missing with sensible defaults
    num_cols = ["income", "loan_amount", "cibil_score", "previous_loans", "missed_emis", "debt_to_income", "age", "dependents"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Map employment types to simple numeric codes
    emp_map = {"salaried": 1, "self-employed": 2, "unemployed": 0, "other": 3}
    df["employment_type"] = df["employment_type"].astype(str).str.lower().map(emp_map).fillna(0).astype(int)

    # Feature engineering: income_to_loan_ratio and emi_burden approximation
    df["income_to_loan_ratio"] = df.apply(lambda r: (r["income"] / (r["loan_amount"] + 1)) if r["loan_amount"] > 0 else r["income"], axis=1)
    df["emi_burden"] = df.apply(lambda r: (r["loan_amount"] * 0.02) / (r["income"] + 1) if r["income"] > 0 else 0, axis=1)

    # Keep a deterministic column order expected by the model
    cols_out = [
        "income",
        "loan_amount",
        "cibil_score",
        "previous_loans",
        "missed_emis",
        "employment_type",
        "debt_to_income",
        "age",
        "dependents",
        "income_to_loan_ratio",
        "emi_burden",
    ]

    return df[cols_out]
