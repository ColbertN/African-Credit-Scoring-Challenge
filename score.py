from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import ID_COL, MODEL_BUNDLE_PATH, REPORTS_DIR, TEST_PATH
from src.monitoring import write_monitoring_reports
from src.scorecard import compute_credit_scores
from src.training import apply_loan_postprocessing, apply_segment_thresholds, predict_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score unseen loans and create submission/scorecard outputs.")
    parser.add_argument("--input-path", type=Path, default=TEST_PATH)
    parser.add_argument("--bundle-path", type=Path, default=MODEL_BUNDLE_PATH)
    parser.add_argument("--submission-path", type=Path, default=Path("submission.csv"))
    parser.add_argument("--scored-path", type=Path, default=REPORTS_DIR / "scored_loans.csv")
    parser.add_argument("--monitoring-dir", type=Path, default=REPORTS_DIR / "monitoring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = joblib.load(args.bundle_path)
    raw = pd.read_csv(args.input_path)

    builder = bundle["feature_builder"]
    feature_frame = builder.transform(raw)
    x = builder.model_features(feature_frame)
    x_transformed = bundle["preprocessor"].transform(x)
    probabilities = predict_ensemble(bundle["models"], x_transformed)

    metadata_cols = ["tbl_loan_id", "country_id", "loan_type"]
    metadata = feature_frame[[col for col in metadata_cols if col in feature_frame.columns]].reset_index(drop=True)
    probabilities = apply_loan_postprocessing(
        metadata,
        probabilities,
        bundle.get("postprocessing_method", "none"),
    )
    predictions = apply_segment_thresholds(
        metadata,
        probabilities,
        float(bundle["threshold"]),
        bundle.get("segment_thresholds"),
    )

    scored_base_cols = [
        ID_COL,
        "customer_id",
        "country_id",
        "tbl_loan_id",
        "lender_id",
        "loan_type",
        "Total_Amount",
        "Total_Amount_to_Repay",
        "duration",
    ]
    scored = raw[[col for col in scored_base_cols if col in raw.columns]].copy()
    scored["probability_default"] = probabilities
    scored["Target"] = predictions.astype(int)
    scored = compute_credit_scores(
        scored,
        credit_history=getattr(builder, "credit_history_", None),
        global_average_loan_amount=getattr(builder, "global_average_loan_amount_", 0.0),
    )

    submission = scored[[ID_COL, "Target"]].copy()
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)

    args.scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scored_path, index=False)

    write_monitoring_reports(
        args.monitoring_dir,
        features=x,
        reference_profile=bundle["monitoring_reference_profile"],
        probabilities=probabilities,
        predictions=predictions,
        scored=scored,
    )

    print(f"Wrote submission to {args.submission_path}")
    print(f"Wrote scorecard output to {args.scored_path}")
    print(f"Wrote monitoring reports to {args.monitoring_dir}")


if __name__ == "__main__":
    main()
