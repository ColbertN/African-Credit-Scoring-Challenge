# African Credit Scoring Challenge

Python workflow for training, scoring, credit score generation, feature importance, and monitoring.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

This trains an ensemble of LightGBM, XGBoost, and CatBoost with engineered financial, date, customer-history, lender, loan, and macroeconomic features. It also applies BorderlineSMOTE inside each CV fold, tunes the F1 threshold from out-of-fold probabilities, learns loan-level post-processing, and saves the trained bundle.

Main artifacts:

- `model/model_bundle.pkl` - full scoring bundle
- `model/lightgbm_model.pkl` - legacy LightGBM model export
- `model/preprocessor_pipeline.pkl` - legacy preprocessor export
- `reports/cv_fold_metrics.csv` - fold metrics
- `reports/postprocessing_results.csv` - loan-level post-processing comparison
- `reports/segment_thresholds.csv` - learned country/loan-type thresholds
- `reports/feature_importance.csv` and `reports/feature_importance.png`
- `reports/training_summary.json`

Current training summary:

```text
OOF F1: 0.8963
Best threshold: 0.7158
Best loan post-processing: max probability per tbl_loan_id
```

Full training can take a while because it fits 4 CV folds plus final versions of all 3 boosted-tree models.

## Score Test Data

```bash
python score.py
```

This reads `data/Test.csv`, loads `model/model_bundle.pkl`, and writes:

- `submission.csv` - Zindi submission with `ID,Target`
- `reports/scored_loans.csv` - probability of default, prediction, credit score, score band, and risk category
- `reports/monitoring/data_drift.csv` - train-vs-score feature drift using PSI
- `reports/monitoring/prediction_summary.csv` - default-rate and score distribution summary
- `reports/monitoring/score_band_distribution.csv`

Current scored test output has 18,594 rows and predicts 680 defaults.

## Credit Score Bands

Scores are capped between 300 and 850:

- `300-579`: Poor
- `580-669`: Fair
- `670-739`: Good
- `740-799`: Very good
- `800-850`: Excellent
