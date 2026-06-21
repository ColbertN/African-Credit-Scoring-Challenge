from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float) + 1e-6
    current = np.asarray(current, dtype=float) + 1e-6
    return float(np.sum((current - reference) * np.log(current / reference)))


def _status_from_psi(value: float) -> str:
    if value >= 0.25:
        return "drift"
    if value >= 0.10:
        return "watch"
    return "stable"


def build_reference_profile(
    features: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    n_bins: int = 10,
    max_categories: int = 25,
) -> dict[str, Any]:
    profile: dict[str, Any] = {"numeric": {}, "categorical": {}}

    for col in numeric_features:
        values = pd.to_numeric(features[col], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.nanquantile(values, quantiles))
        if len(edges) < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts = pd.cut(values, bins=edges, include_lowest=True).value_counts(sort=False, normalize=True)
        profile["numeric"][col] = {
            "edges": edges.tolist(),
            "distribution": counts.to_numpy().tolist(),
            "missing_rate": float(features[col].isna().mean()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    for col in categorical_features:
        values = features[col].astype(str).fillna("Unknown")
        top_categories = values.value_counts().head(max_categories).index.tolist()
        bucketed = values.where(values.isin(top_categories), "__OTHER__")
        distribution = bucketed.value_counts(normalize=True)
        categories = distribution.index.tolist()
        profile["categorical"][col] = {
            "categories": categories,
            "distribution": distribution.reindex(categories).fillna(0).to_numpy().tolist(),
            "missing_rate": float(features[col].isna().mean()),
        }

    return profile


def evaluate_data_drift(features: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for col, ref in profile.get("numeric", {}).items():
        if col not in features.columns:
            continue
        values = pd.to_numeric(features[col], errors="coerce")
        edges = np.asarray(ref["edges"], dtype=float)
        current_dist = (
            pd.cut(values.dropna(), bins=edges, include_lowest=True)
            .value_counts(sort=False, normalize=True)
            .to_numpy()
        )
        reference_dist = np.asarray(ref["distribution"], dtype=float)
        if len(current_dist) != len(reference_dist):
            current_dist = np.resize(current_dist, len(reference_dist))
        psi_value = _psi(reference_dist, current_dist)
        rows.append(
            {
                "feature": col,
                "feature_type": "numeric",
                "psi": psi_value,
                "status": _status_from_psi(psi_value),
                "reference_missing_rate": ref.get("missing_rate", 0.0),
                "current_missing_rate": float(values.isna().mean()),
                "reference_mean": ref.get("mean"),
                "current_mean": float(values.mean(skipna=True)),
            }
        )

    for col, ref in profile.get("categorical", {}).items():
        if col not in features.columns:
            continue
        categories = ref["categories"]
        values = features[col].astype(str).fillna("Unknown")
        bucketed = values.where(values.isin(categories), "__OTHER__")
        current_dist = bucketed.value_counts(normalize=True).reindex(categories).fillna(0).to_numpy()
        reference_dist = np.asarray(ref["distribution"], dtype=float)
        psi_value = _psi(reference_dist, current_dist)
        rows.append(
            {
                "feature": col,
                "feature_type": "categorical",
                "psi": psi_value,
                "status": _status_from_psi(psi_value),
                "reference_missing_rate": ref.get("missing_rate", 0.0),
                "current_missing_rate": float(features[col].isna().mean()),
                "reference_mean": np.nan,
                "current_mean": np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def summarize_predictions(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    scored: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rows": int(len(probabilities)),
                "predicted_default_rate": float(np.mean(predictions)),
                "mean_probability_default": float(np.mean(probabilities)),
                "p05_probability_default": float(np.quantile(probabilities, 0.05)),
                "p50_probability_default": float(np.quantile(probabilities, 0.50)),
                "p95_probability_default": float(np.quantile(probabilities, 0.95)),
                "mean_credit_score": float(scored["credit_score"].mean()),
                "p05_credit_score": float(scored["credit_score"].quantile(0.05)),
                "p50_credit_score": float(scored["credit_score"].quantile(0.50)),
                "p95_credit_score": float(scored["credit_score"].quantile(0.95)),
            }
        ]
    )


def write_monitoring_reports(
    output_dir: str | Path,
    features: pd.DataFrame,
    reference_profile: dict[str, Any],
    probabilities: np.ndarray,
    predictions: np.ndarray,
    scored: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    drift = evaluate_data_drift(features, reference_profile)
    drift.to_csv(output_dir / "data_drift.csv", index=False)

    summary = summarize_predictions(probabilities, predictions, scored)
    summary.to_csv(output_dir / "prediction_summary.csv", index=False)

    band_distribution = (
        scored["score_band"]
        .value_counts(dropna=False)
        .rename_axis("score_band")
        .reset_index(name="rows")
    )
    band_distribution["share"] = band_distribution["rows"] / max(len(scored), 1)
    band_distribution.to_csv(output_dir / "score_band_distribution.csv", index=False)
