from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"
REPORTS_DIR = ROOT_DIR / "reports"

TRAIN_PATH = DATA_DIR / "Train.csv"
TEST_PATH = DATA_DIR / "Test.csv"
ECONOMIC_INDICATORS_PATH = DATA_DIR / "economic_indicators.csv"

MODEL_BUNDLE_PATH = MODEL_DIR / "model_bundle.pkl"
LEGACY_MODEL_PATH = MODEL_DIR / "lightgbm_model.pkl"
LEGACY_PREPROCESSOR_PATH = MODEL_DIR / "preprocessor_pipeline.pkl"

ID_COL = "ID"
TARGET_COL = "target"
DATE_COLS = ["disbursement_date", "due_date"]
