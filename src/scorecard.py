from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_MIN = 300
SCORE_MAX = 850


def score_band(score: float) -> str:
    if score < 580:
        return "Poor"
    if score < 670:
        return "Fair"
    if score < 740:
        return "Good"
    if score < 800:
        return "Very good"
    return "Excellent"


def risk_category(score: float) -> str:
    if score < 580:
        return "Very high risk"
    if score < 670:
        return "High risk"
    if score < 740:
        return "Medium risk"
    if score < 800:
        return "Low risk"
    return "Very low risk"


def compute_credit_scores(
    scored: pd.DataFrame,
    credit_history: pd.DataFrame | None,
    global_average_loan_amount: float,
) -> pd.DataFrame:
    output = scored.copy()
    if credit_history is None or credit_history.empty:
        history = pd.DataFrame({"customer_id": output["customer_id"].unique()})
        history["loan_history_length"] = 0
        history["default_payment_count"] = 0
        history["historical_default_rate"] = 0.0
        history["average_loan_amount"] = global_average_loan_amount
    else:
        history = credit_history.copy()

    output = output.merge(history, on="customer_id", how="left")
    output["loan_history_length"] = output["loan_history_length"].fillna(0)
    output["default_payment_count"] = output["default_payment_count"].fillna(0)
    output["historical_default_rate"] = output["historical_default_rate"].fillna(0)
    output["average_loan_amount"] = output["average_loan_amount"].fillna(global_average_loan_amount)

    loan_amount = pd.to_numeric(output["Total_Amount"], errors="coerce").fillna(global_average_loan_amount)
    loan_to_history_amount = loan_amount / (output["average_loan_amount"].replace(0, np.nan) + 1e-6)
    loan_to_history_amount = loan_to_history_amount.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    probability = output["probability_default"].clip(0, 1)
    raw_score = (
        SCORE_MAX
        - 430 * probability
        - 24 * output["default_payment_count"].clip(upper=20)
        - 95 * output["historical_default_rate"].clip(0, 1)
        - 35 * loan_to_history_amount.clip(lower=0, upper=8)
        + 5 * output["loan_history_length"].clip(upper=30)
    )
    output["credit_score"] = raw_score.clip(SCORE_MIN, SCORE_MAX).round().astype(int)
    output["score_band"] = output["credit_score"].map(score_band)
    output["risk_category"] = output["credit_score"].map(risk_category)
    return output
