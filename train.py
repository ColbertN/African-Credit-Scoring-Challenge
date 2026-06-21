from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    ECONOMIC_INDICATORS_PATH,
    LEGACY_MODEL_PATH,
    LEGACY_PREPROCESSOR_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_DIR,
    REPORTS_DIR,
    TARGET_COL,
    TRAIN_PATH,
)
from src.features import FeatureBuilder, feature_lists_to_dict
from src.monitoring import build_reference_profile
from src.training import (
    build_preprocessor,
    compute_feature_importance,
    cross_validate_ensemble,
    get_feature_names,
    save_bundle,
    save_feature_importance_plot,
    train_final_ensemble,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the African credit default ensemble.")
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--economic-path", type=Path, default=ECONOMIC_INDICATORS_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smote-strategy", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_path)
    builder = FeatureBuilder(economic_indicators_path=args.economic_path)
    feature_frame = builder.fit_transform(train)
    x = builder.model_features(feature_frame)
    y = train[TARGET_COL].astype(int)

    metadata_cols = ["ID", "tbl_loan_id", "country_id", "loan_type", "customer_id", "Total_Amount"]
    metadata = feature_frame[[col for col in metadata_cols if col in feature_frame.columns]].copy()
    metadata = metadata.reset_index(drop=True)

    preprocessor = build_preprocessor(builder.numeric_features_, builder.categorical_features_)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    cv_result = cross_validate_ensemble(
        x=x,
        y=y,
        metadata=metadata,
        preprocessor=preprocessor,
        cv=cv,
        sampling_strategy=args.smote_strategy,
        random_state=args.random_state,
    )

    fitted_preprocessor, models, _, _ = train_final_ensemble(
        x=x,
        y=y,
        preprocessor=preprocessor,
        sampling_strategy=args.smote_strategy,
        random_state=args.random_state,
    )
    feature_names = get_feature_names(fitted_preprocessor, builder.feature_columns_)
    importance = compute_feature_importance(models, feature_names)

    reference_profile = build_reference_profile(
        x,
        numeric_features=builder.numeric_features_,
        categorical_features=builder.categorical_features_,
    )

    bundle = {
        "feature_builder": builder,
        "preprocessor": fitted_preprocessor,
        "models": models,
        "feature_names": feature_names,
        "threshold": cv_result["threshold"],
        "postprocessing_method": cv_result["best_postprocessing_method"],
        "segment_thresholds": cv_result["segment_thresholds"],
        "monitoring_reference_profile": reference_profile,
        "feature_schema": feature_lists_to_dict(builder),
        "training_summary": {
            "n_rows": int(len(train)),
            "target_rate": float(y.mean()),
            "oof_f1": float(cv_result["final_f1"]),
            "threshold": float(cv_result["threshold"]),
            "postprocessing_method": cv_result["best_postprocessing_method"],
            "smote_strategy": float(args.smote_strategy),
            "n_splits": int(args.n_splits),
            "random_state": int(args.random_state),
        },
    }

    save_bundle(bundle, MODEL_BUNDLE_PATH)
    joblib.dump(fitted_preprocessor, LEGACY_PREPROCESSOR_PATH)
    joblib.dump(models["lightgbm"], LEGACY_MODEL_PATH)

    cv_result["fold_metrics"].to_csv(args.reports_dir / "cv_fold_metrics.csv", index=False)
    cv_result["postprocessing_results"].to_csv(args.reports_dir / "postprocessing_results.csv", index=False)
    cv_result["segment_thresholds"].to_csv(args.reports_dir / "segment_thresholds.csv", index=False)
    importance.to_csv(args.reports_dir / "feature_importance.csv", index=False)
    save_feature_importance_plot(importance, args.reports_dir / "feature_importance.png")

    with open(args.reports_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(bundle["training_summary"], f, indent=2)

    print("Training complete")
    print(json.dumps(bundle["training_summary"], indent=2))


if __name__ == "__main__":
    main()
