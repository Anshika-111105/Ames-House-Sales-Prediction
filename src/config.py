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
