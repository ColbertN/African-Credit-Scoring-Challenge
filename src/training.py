from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import BorderlineSMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from xgboost import XGBClassifier


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_models(scale_pos_weight: float, random_state: int) -> dict[str, Any]:
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=450,
            max_depth=8,
            learning_rate=0.055,
            num_leaves=70,
            feature_fraction=0.72,
            bagging_fraction=0.78,
            bagging_freq=1,
            lambda_l1=0.05,
            lambda_l2=0.85,
            min_child_samples=25,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=420,
            max_depth=7,
            learning_rate=0.065,
            subsample=0.9,
            colsample_bytree=0.72,
            gamma=0.01,
            min_child_weight=1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.065,
            loss_function="Logloss",
            scale_pos_weight=scale_pos_weight,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def optimize_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 199), probabilities])
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def apply_loan_postprocessing(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "none" or "tbl_loan_id" not in metadata.columns:
        return probabilities

    frame = metadata[["tbl_loan_id"]].copy()
    frame["_probability"] = probabilities
    if method == "mean":
        return frame.groupby("tbl_loan_id")["_probability"].transform("mean").to_numpy()
    if method == "max":
        return frame.groupby("tbl_loan_id")["_probability"].transform("max").to_numpy()
    raise ValueError(f"Unsupported loan post-processing method: {method}")


def evaluate_postprocessing(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for method in ["none", "mean", "max"]:
        adjusted = apply_loan_postprocessing(metadata, probabilities, method)
        threshold, score = optimize_f1_threshold(y_true, adjusted)
        rows.append(
            {
                "postprocessing_method": method,
                "threshold": threshold,
                "f1_score": score,
            }
        )
    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    best = results.iloc[0].to_dict()
    best["all_results"] = results
    return best


def learn_segment_thresholds(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    global_threshold: float,
    min_rows: int = 250,
    min_positive: int = 5,
) -> tuple[pd.DataFrame, float]:
    if not {"country_id", "loan_type"}.issubset(metadata.columns):
        return pd.DataFrame(), f1_score(y_true, probabilities >= global_threshold, zero_division=0)

    baseline_preds = (probabilities >= global_threshold).astype(int)
    baseline_f1 = f1_score(y_true, baseline_preds, zero_division=0)
    candidate_preds = baseline_preds.copy()
    rows = []

    frame = metadata[["country_id", "loan_type"]].copy()
    frame["_y"] = y_true
    frame["_probability"] = probabilities
    for (country, loan_type), group in frame.groupby(["country_id", "loan_type"], dropna=False):
        positives = int(group["_y"].sum())
        if len(group) < min_rows or positives < min_positive:
            continue
        threshold, group_f1 = optimize_f1_threshold(
            group["_y"].to_numpy(), group["_probability"].to_numpy()
        )
        idx = group.index.to_numpy()
        candidate_preds[idx] = (group["_probability"].to_numpy() >= threshold).astype(int)
        rows.append(
            {
                "country_id": str(country),
                "loan_type": str(loan_type),
                "threshold": threshold,
                "rows": int(len(group)),
                "positives": positives,
                "segment_f1": group_f1,
            }
        )

    candidate_f1 = f1_score(y_true, candidate_preds, zero_division=0)
    if candidate_f1 <= baseline_f1:
        return pd.DataFrame(), baseline_f1
    return pd.DataFrame(rows), float(candidate_f1)


def apply_segment_thresholds(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    global_threshold: float,
    segment_thresholds: pd.DataFrame | None,
) -> np.ndarray:
    thresholds = np.full(len(probabilities), global_threshold, dtype=float)
    if segment_thresholds is None or segment_thresholds.empty:
        return (probabilities >= thresholds).astype(int)

    lookup = {
        (str(row["country_id"]), str(row["loan_type"])): float(row["threshold"])
        for _, row in segment_thresholds.iterrows()
    }
    for pos, row in enumerate(metadata[["country_id", "loan_type"]].itertuples(index=False)):
        thresholds[pos] = lookup.get((str(row.country_id), str(row.loan_type)), global_threshold)
    return (probabilities >= thresholds).astype(int)


def _scale_pos_weight(y: np.ndarray) -> float:
    positives = max(float(np.sum(y == 1)), 1.0)
    negatives = max(float(np.sum(y == 0)), 1.0)
    return max(1.0, negatives / positives)


def _resample_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling_strategy <= 0:
        return x_train, y_train
    minority_count = int(np.sum(y_train == 1))
    majority_count = int(np.sum(y_train == 0))
    if minority_count < 8:
        return x_train, y_train
    if majority_count <= 0 or minority_count / max(majority_count, 1) >= sampling_strategy:
        return x_train, y_train
    sampler = BorderlineSMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=min(5, minority_count - 1),
    )
    return sampler.fit_resample(x_train, y_train)


def predict_ensemble(models: dict[str, Any], x_transformed: np.ndarray, weights: dict[str, float] | None = None) -> np.ndarray:
    probabilities = []
    used_weights = []
    for name, model in models.items():
        probabilities.append(model.predict_proba(x_transformed)[:, 1])
        used_weights.append(1.0 if weights is None else weights.get(name, 1.0))
    probabilities = np.vstack(probabilities)
    used_weights = np.array(used_weights, dtype=float)
    used_weights = used_weights / used_weights.sum()
    return np.average(probabilities, axis=0, weights=used_weights)


def cross_validate_ensemble(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    preprocessor: ColumnTransformer,
    cv,
    sampling_strategy: float,
    random_state: int,
) -> dict[str, Any]:
    y_array = y.to_numpy()
    oof_probabilities = np.zeros(len(x), dtype=float)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
        fold_preprocessor = clone(preprocessor)
        x_train = fold_preprocessor.fit_transform(x.iloc[train_idx])
        x_valid = fold_preprocessor.transform(x.iloc[valid_idx])
        y_train = y_array[train_idx]
        y_valid = y_array[valid_idx]

        x_resampled, y_resampled = _resample_training_data(
            x_train, y_train, sampling_strategy=sampling_strategy, random_state=random_state + fold
        )
        scale_pos_weight = _scale_pos_weight(y_resampled)
        models = make_models(scale_pos_weight=scale_pos_weight, random_state=random_state + fold)
        for model in models.values():
            model.fit(x_resampled, y_resampled)

        fold_prob = predict_ensemble(models, x_valid)
        oof_probabilities[valid_idx] = fold_prob
        fold_threshold, fold_f1 = optimize_f1_threshold(y_valid, fold_prob)
        fold_preds = (fold_prob >= fold_threshold).astype(int)
        fold_rows.append(
            {
                "fold": fold,
                "threshold": fold_threshold,
                "f1_score": fold_f1,
                "precision": precision_score(y_valid, fold_preds, zero_division=0),
                "recall": recall_score(y_valid, fold_preds, zero_division=0),
                "roc_auc": roc_auc_score(y_valid, fold_prob),
                "scale_pos_weight": scale_pos_weight,
                "train_rows_after_smote": int(len(y_resampled)),
            }
        )

    post = evaluate_postprocessing(metadata, y_array, oof_probabilities)
    adjusted_oof = apply_loan_postprocessing(
        metadata, oof_probabilities, post["postprocessing_method"]
    )
    segment_thresholds, segment_f1 = learn_segment_thresholds(
        metadata, y_array, adjusted_oof, float(post["threshold"])
    )
    final_f1 = max(float(post["f1_score"]), segment_f1)

    return {
        "oof_probabilities": oof_probabilities,
        "adjusted_oof_probabilities": adjusted_oof,
        "fold_metrics": pd.DataFrame(fold_rows),
        "postprocessing_results": post["all_results"],
        "best_postprocessing_method": post["postprocessing_method"],
        "threshold": float(post["threshold"]),
        "segment_thresholds": segment_thresholds,
        "segment_threshold_f1": segment_f1,
        "final_f1": final_f1,
    }


def train_final_ensemble(
    x: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    sampling_strategy: float,
    random_state: int,
) -> tuple[ColumnTransformer, dict[str, Any], np.ndarray, np.ndarray]:
    fitted_preprocessor = deepcopy(preprocessor)
    x_transformed = fitted_preprocessor.fit_transform(x)
    x_resampled, y_resampled = _resample_training_data(
        x_transformed, y.to_numpy(), sampling_strategy=sampling_strategy, random_state=random_state
    )
    scale_pos_weight = _scale_pos_weight(y_resampled)
    models = make_models(scale_pos_weight=scale_pos_weight, random_state=random_state)
    for model in models.values():
        model.fit(x_resampled, y_resampled)
    return fitted_preprocessor, models, x_transformed, y_resampled


def get_feature_names(preprocessor: ColumnTransformer, fallback: list[str]) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return fallback


def compute_feature_importance(models: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for model_name, model in models.items():
        importance = getattr(model, "feature_importances_", None)
        if importance is None:
            continue
        importance = np.asarray(importance, dtype=float)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        for feature, value in zip(feature_names, importance):
            rows.append({"model": model_name, "feature": feature, "importance": value})

    if not rows:
        return pd.DataFrame(columns=["feature", "mean_importance"])

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("feature", as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "mean_importance"})
        .sort_values("mean_importance", ascending=False)
    )
    return summary


def save_feature_importance_plot(importance: pd.DataFrame, path: str | Path, top_n: int = 30) -> None:
    if importance.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(top_n).sort_values("mean_importance")
    plt.figure(figsize=(9, max(5, len(top) * 0.25)))
    plt.barh(top["feature"], top["mean_importance"])
    plt.xlabel("Mean normalized importance")
    plt.title(f"Top {min(top_n, len(importance))} feature importances")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
