import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"


MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.pkl"

OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.txt"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

# Required project deliverables
MODEL_COMPARISON_CSV = OUTPUT_DIR / "model_comparison.csv"
PREDICTION_SAMPLES_CSV = OUTPUT_DIR / "prediction_samples.csv"
RESIDUAL_PLOT_PNG = OUTPUT_DIR / "residual_plot.png"
BEST_PIPELINE_JOBLIB = OUTPUT_DIR / "best_house_price_pipeline.joblib"
BUSINESS_INTERPRETATION_MD = OUTPUT_DIR / "business_interpretation.md"


TARGET_COL = "SalePrice"
ID_COL = "Id"


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
RIDGE_PARAM_GRID = {
    "model__alpha": [0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
}

LASSO_PARAM_GRID = {
    "model__alpha": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 50.0]
}

# URLs for dataset download (GitHub mirror of Kaggle House Prices competition)
TRAIN_URL = "https://raw.githubusercontent.com/sidhantagar/Kaggle-House-Prices/master/train.csv"
TEST_URL = "https://raw.githubusercontent.com/sidhantagar/Kaggle-House-Prices/master/test.csv"
