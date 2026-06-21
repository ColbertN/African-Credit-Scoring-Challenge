from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ECONOMIC_INDICATORS_PATH, ID_COL, TARGET_COL


EPSILON = 1e-6


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / (denominator + EPSILON)).replace([np.inf, -np.inf], np.nan)


def load_economic_indicators(path: str | Path = ECONOMIC_INDICATORS_PATH) -> pd.DataFrame:
    """Return one row per country/year with indicators as columns."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["country_id", "macro_year"])

    raw = pd.read_csv(path)
    year_cols = [col for col in raw.columns if col.startswith("YR")]
    if raw.empty or not year_cols:
        return pd.DataFrame(columns=["country_id", "macro_year"])

    long = raw.melt(
        id_vars=["Country", "Indicator"],
        value_vars=year_cols,
        var_name="macro_year",
        value_name="indicator_value",
    )
    long["macro_year"] = long["macro_year"].str.replace("YR", "", regex=False).astype(int)
    long["indicator_value"] = pd.to_numeric(long["indicator_value"], errors="coerce")

    econ = (
        long.pivot_table(
            index=["Country", "macro_year"],
            columns="Indicator",
            values="indicator_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"Country": "country_id"})
    )
    econ.columns.name = None
    indicator_cols = [col for col in econ.columns if col not in ["country_id", "macro_year"]]
    econ = econ.sort_values(["country_id", "macro_year"])
    econ[indicator_cols] = econ.groupby("country_id", dropna=False)[indicator_cols].ffill().bfill()
    econ[indicator_cols] = econ[indicator_cols].fillna(econ[indicator_cols].median(numeric_only=True))
    return econ


@dataclass
class FeatureBuilder:
    economic_indicators_path: str | Path = ECONOMIC_INDICATORS_PATH
    identifier_features: list[str] = field(
        default_factory=lambda: ["customer_id", "tbl_loan_id", "lender_id"]
    )

    economic_features_: pd.DataFrame | None = None
    min_macro_year_: int | None = None
    max_macro_year_: int | None = None
    customer_aggregates_: pd.DataFrame | None = None
    lender_aggregates_: pd.DataFrame | None = None
    country_type_aggregates_: pd.DataFrame | None = None
    aggregate_fill_values_: dict[str, float] = field(default_factory=dict)
    numeric_fill_values_: dict[str, float] = field(default_factory=dict)
    credit_history_: pd.DataFrame | None = None
    global_average_loan_amount_: float = 0.0
    feature_columns_: list[str] = field(default_factory=list)
    numeric_features_: list[str] = field(default_factory=list)
    categorical_features_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.economic_features_ = load_economic_indicators(self.economic_indicators_path)
        if not self.economic_features_.empty:
            self.min_macro_year_ = int(self.economic_features_["macro_year"].min())
            self.max_macro_year_ = int(self.economic_features_["macro_year"].max())

        base = self._create_base_features(df)
        self._fit_aggregate_tables(base)
        self._fit_credit_history(df)

        engineered = self._add_aggregate_features(base)
        engineered = self._add_batch_loan_features(engineered)
        self._infer_feature_schema(engineered)
        self._fit_fill_values(engineered)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns_:
            raise RuntimeError("FeatureBuilder must be fitted before transform.")

        engineered = self._create_base_features(df)
        engineered = self._add_aggregate_features(engineered)
        engineered = self._add_batch_loan_features(engineered)
        engineered = self._finalize_feature_frame(engineered)
        return engineered

    def model_features(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_columns_ if col not in feature_frame.columns]
        if missing:
            feature_frame = feature_frame.copy()
            for col in missing:
                feature_frame[col] = self.numeric_fill_values_.get(col, "Unknown")
        return feature_frame[self.feature_columns_]

    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        for col in ["Total_Amount", "Total_Amount_to_Repay", "Amount_Funded_By_Lender",
                    "Lender_portion_Funded", "Lender_portion_to_be_repaid", "duration"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        if {"Total_Amount_to_Repay", "Lender_portion_to_be_repaid"}.issubset(data.columns):
            repay_is_zero = data["Total_Amount_to_Repay"].fillna(0).eq(0)
            data.loc[repay_is_zero, "Total_Amount_to_Repay"] = data.loc[
                repay_is_zero, "Lender_portion_to_be_repaid"
            ]

        data["disbursement_date"] = pd.to_datetime(data["disbursement_date"], errors="coerce")
        data["due_date"] = pd.to_datetime(data["due_date"], errors="coerce")
        data["loan_term_days"] = (data["due_date"] - data["disbursement_date"]).dt.days
        data["duration_term_gap"] = data["duration"] - data["loan_term_days"]

        for prefix, col in [
            ("disbursement", "disbursement_date"),
            ("due", "due_date"),
        ]:
            data[f"{prefix}_year"] = data[col].dt.year
            data[f"{prefix}_month"] = data[col].dt.month
            data[f"{prefix}_day"] = data[col].dt.day
            data[f"{prefix}_dayofweek"] = data[col].dt.dayofweek
            data[f"{prefix}_quarter"] = data[col].dt.quarter
            data[f"{prefix}_is_month_end"] = data[col].dt.is_month_end.astype("float")

        data["is_repeat_loan"] = (
            data.get("New_versus_Repeat", pd.Series(index=data.index, dtype=object))
            .astype(str)
            .str.lower()
            .str.contains("repeat", na=False)
            .astype(float)
        )
        data["duration_bucket"] = pd.cut(
            data["duration"],
            bins=[-np.inf, 7, 14, 30, 60, 90, np.inf],
            labels=["0_7", "8_14", "15_30", "31_60", "61_90", "90_plus"],
        ).astype(str)

        data["repayment_ratio"] = safe_divide(data["Total_Amount_to_Repay"], data["Total_Amount"])
        data["repay_minus_amount"] = data["Total_Amount_to_Repay"] - data["Total_Amount"]
        data["repay_spread_pct"] = safe_divide(data["repay_minus_amount"], data["Total_Amount"])
        data["amount_due_per_day"] = safe_divide(data["Total_Amount_to_Repay"], data["duration"])
        data["amount_per_day"] = safe_divide(data["Total_Amount"], data["duration"])
        data["lender_repay_per_day"] = safe_divide(data["Lender_portion_to_be_repaid"], data["duration"])
        data["lender_funded_ratio"] = safe_divide(data["Amount_Funded_By_Lender"], data["Total_Amount"])
        data["lender_repayment_ratio"] = safe_divide(
            data["Lender_portion_to_be_repaid"], data["Amount_Funded_By_Lender"]
        )
        data["lender_portion_gap"] = (
            data["Lender_portion_to_be_repaid"] - data["Amount_Funded_By_Lender"]
        )
        data["lender_portion_gap_pct"] = safe_divide(
            data["lender_portion_gap"], data["Amount_Funded_By_Lender"]
        )
        data["amount_duration_product"] = data["Total_Amount"] * data["duration"]

        for col in [
            "Total_Amount",
            "Total_Amount_to_Repay",
            "Amount_Funded_By_Lender",
            "Lender_portion_to_be_repaid",
            "duration",
            "amount_due_per_day",
        ]:
            data[f"log_{col}"] = np.log1p(data[col].clip(lower=0))

        data = self._merge_economic_indicators(data)
        data = data.drop(columns=["disbursement_date", "due_date"], errors="ignore")
        return data

    def _merge_economic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        econ = self.economic_features_
        if econ is None or econ.empty:
            data["macro_year"] = data["disbursement_year"]
            return data

        min_year = self.min_macro_year_ or int(econ["macro_year"].min())
        max_year = self.max_macro_year_ or int(econ["macro_year"].max())
        data["macro_year"] = data["disbursement_year"].clip(lower=min_year, upper=max_year)
        data["macro_year"] = data["macro_year"].fillna(max_year).astype(int)
        data = data.merge(econ, on=["country_id", "macro_year"], how="left")
        data["macro_year_gap"] = data["disbursement_year"] - data["macro_year"]
        return data

    def _fit_aggregate_tables(self, base: pd.DataFrame) -> None:
        self.global_average_loan_amount_ = float(base["Total_Amount"].median())

        self.customer_aggregates_ = (
            base.groupby("customer_id", dropna=False)
            .agg(
                customer_seen_loans=("tbl_loan_id", "nunique"),
                customer_seen_lenders=("lender_id", "nunique"),
                customer_seen_loan_types=("loan_type", "nunique"),
                customer_amount_mean=("Total_Amount", "mean"),
                customer_amount_median=("Total_Amount", "median"),
                customer_amount_max=("Total_Amount", "max"),
                customer_amount_std=("Total_Amount", "std"),
                customer_repay_mean=("Total_Amount_to_Repay", "mean"),
                customer_repay_median=("Total_Amount_to_Repay", "median"),
                customer_duration_mean=("duration", "mean"),
                customer_repayment_ratio_mean=("repayment_ratio", "mean"),
                customer_repeat_rate=("is_repeat_loan", "mean"),
            )
            .reset_index()
        )

        self.lender_aggregates_ = (
            base.groupby("lender_id", dropna=False)
            .agg(
                lender_seen_loans=("tbl_loan_id", "nunique"),
                lender_seen_customers=("customer_id", "nunique"),
                lender_amount_mean=("Total_Amount", "mean"),
                lender_repay_mean=("Total_Amount_to_Repay", "mean"),
                lender_duration_mean=("duration", "mean"),
                lender_funded_ratio_mean=("lender_funded_ratio", "mean"),
            )
            .reset_index()
        )

        self.country_type_aggregates_ = (
            base.groupby(["country_id", "loan_type"], dropna=False)
            .agg(
                country_type_seen_loans=("tbl_loan_id", "nunique"),
                country_type_amount_mean=("Total_Amount", "mean"),
                country_type_repay_mean=("Total_Amount_to_Repay", "mean"),
                country_type_duration_mean=("duration", "mean"),
                country_type_repayment_ratio_mean=("repayment_ratio", "mean"),
            )
            .reset_index()
        )

        aggregate_frames = [
            self.customer_aggregates_,
            self.lender_aggregates_,
            self.country_type_aggregates_,
        ]
        self.aggregate_fill_values_ = {}
        for frame in aggregate_frames:
            key_cols = {"customer_id", "lender_id", "country_id", "loan_type"}
            for col in [c for c in frame.columns if c not in key_cols]:
                self.aggregate_fill_values_[col] = float(frame[col].median(skipna=True))

    def _fit_credit_history(self, df: pd.DataFrame) -> None:
        if TARGET_COL not in df.columns:
            self.credit_history_ = pd.DataFrame(
                columns=[
                    "customer_id",
                    "loan_history_length",
                    "default_payment_count",
                    "historical_default_rate",
                    "average_loan_amount",
                ]
            )
            return

        history = (
            df.groupby("customer_id", dropna=False)
            .agg(
                loan_history_length=("tbl_loan_id", "nunique"),
                default_payment_count=(TARGET_COL, "sum"),
                historical_default_rate=(TARGET_COL, "mean"),
                average_loan_amount=("Total_Amount", "mean"),
            )
            .reset_index()
        )
        self.credit_history_ = history

    def _add_aggregate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._merge_aggregate(data, self.customer_aggregates_, ["customer_id"])
        data = self._merge_aggregate(data, self.lender_aggregates_, ["lender_id"])
        data = self._merge_aggregate(data, self.country_type_aggregates_, ["country_id", "loan_type"])

        data["amount_vs_customer_mean"] = data["Total_Amount"] - data["customer_amount_mean"]
        data["repay_vs_customer_mean"] = data["Total_Amount_to_Repay"] - data["customer_repay_mean"]
        data["amount_to_customer_mean_ratio"] = safe_divide(
            data["Total_Amount"], data["customer_amount_mean"]
        )
        data["repay_to_customer_mean_ratio"] = safe_divide(
            data["Total_Amount_to_Repay"], data["customer_repay_mean"]
        )
        data["amount_vs_lender_mean"] = data["Total_Amount"] - data["lender_amount_mean"]
        data["amount_to_country_type_mean_ratio"] = safe_divide(
            data["Total_Amount"], data["country_type_amount_mean"]
        )
        return data

    def _merge_aggregate(self, data: pd.DataFrame, table: pd.DataFrame | None, keys: list[str]) -> pd.DataFrame:
        if table is None or table.empty:
            return data
        data = data.merge(table, on=keys, how="left")
        for col in [c for c in table.columns if c not in keys]:
            fill_value = self.aggregate_fill_values_.get(col, 0.0)
            data[col] = data[col].fillna(fill_value)
        return data

    def _add_batch_loan_features(self, data: pd.DataFrame) -> pd.DataFrame:
        batch = (
            data.groupby("tbl_loan_id", dropna=False)
            .agg(
                batch_loan_row_count=(ID_COL, "size"),
                batch_loan_lender_count=("lender_id", "nunique"),
                batch_loan_funded_sum=("Amount_Funded_By_Lender", "sum"),
                batch_loan_repay_sum=("Lender_portion_to_be_repaid", "sum"),
            )
            .reset_index()
        )
        data = data.merge(batch, on="tbl_loan_id", how="left")
        data["batch_lender_funding_share"] = safe_divide(
            data["Amount_Funded_By_Lender"], data["batch_loan_funded_sum"]
        )
        data["batch_lender_repay_share"] = safe_divide(
            data["Lender_portion_to_be_repaid"], data["batch_loan_repay_sum"]
        )
        return data

    def _infer_feature_schema(self, engineered: pd.DataFrame) -> None:
        non_features = {ID_COL, TARGET_COL}
        self.feature_columns_ = [col for col in engineered.columns if col not in non_features]

        categorical_candidates = set(self.identifier_features + ["country_id", "loan_type", "New_versus_Repeat", "duration_bucket"])
        self.categorical_features_ = [
            col
            for col in self.feature_columns_
            if col in categorical_candidates
            or engineered[col].dtype == "object"
            or str(engineered[col].dtype).startswith("category")
        ]
        self.numeric_features_ = [
            col for col in self.feature_columns_ if col not in self.categorical_features_
        ]

    def _fit_fill_values(self, engineered: pd.DataFrame) -> None:
        numeric = engineered[self.numeric_features_].replace([np.inf, -np.inf], np.nan)
        fill_values = numeric.median(numeric_only=True).fillna(0.0)
        self.numeric_fill_values_ = {col: float(fill_values.get(col, 0.0)) for col in self.numeric_features_}

    def _finalize_feature_frame(self, engineered: pd.DataFrame) -> pd.DataFrame:
        engineered = engineered.copy()
        engineered = engineered.replace([np.inf, -np.inf], np.nan)

        for col in self.categorical_features_:
            if col not in engineered.columns:
                engineered[col] = "Unknown"
            engineered[col] = engineered[col].astype("string").fillna("Unknown").astype(str)

        for col in self.numeric_features_:
            if col not in engineered.columns:
                engineered[col] = self.numeric_fill_values_.get(col, 0.0)
            engineered[col] = pd.to_numeric(engineered[col], errors="coerce")
            engineered[col] = engineered[col].fillna(self.numeric_fill_values_.get(col, 0.0))

        return engineered


def feature_lists_to_dict(builder: FeatureBuilder) -> dict[str, Iterable[str]]:
    return {
        "feature_columns": builder.feature_columns_,
        "numeric_features": builder.numeric_features_,
        "categorical_features": builder.categorical_features_,
    }
