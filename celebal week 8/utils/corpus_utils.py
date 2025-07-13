import pandas as pd

def build_corpus(df, approved_only=False, graduate_only=False):
    df = df.copy()

    if approved_only:
        df = df[df["Loan_Status"] == "Y"]

    if graduate_only:
        df = df[df["Education"] == "Graduate"]

    df = df.fillna("Unknown")

    def row_to_text(row):
        return f"Applicant {row['Gender']} aged {row['ApplicantIncome']} from {row['Property_Area']} with {row['Education']} education applied for a {row['LoanAmount']} loan. Status: {row['Loan_Status']}"

    df["text"] = df.apply(row_to_text, axis=1)
    return df
