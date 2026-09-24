import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.config as config
import src.utils as utils
import src.regression_pipeline as rp
import src.kpi_metrics as kpi_metrics
import src.recommendation_engine as rec_engine

# ============================================================
# 1. Data Validation Tests
# ============================================================
def test_data_files_exist():
    """Verify that train and test datasets exist."""
    assert config.TRAIN_DATA_PATH.exists(), f"Train dataset missing at {config.TRAIN_DATA_PATH}"
    assert config.TEST_DATA_PATH.exists(), f"Test dataset missing at {config.TEST_DATA_PATH}"

def test_train_data_schema():
    """Ensure training data contains critical features and target variable."""
    df = utils.load_data(config.TRAIN_DATA_PATH, is_train=True)
    required_cols = [config.TARGET_COL, "GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "Neighborhood"]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from train dataset."
    assert df[config.TARGET_COL].isnull().sum() == 0, "Target variable contains null values."
    assert (df[config.TARGET_COL] > 0).all(), "Target variable contains non-positive values."

# ============================================================
# 2. Custom Feature Engineering Tests
# ============================================================
def test_feature_engineer_transformations():
    """Test that custom FeatureEngineer creates expected composite features correctly."""
    sample_df = pd.DataFrame({
        "YrSold": [2010, 2008],
        "YearBuilt": [2000, 2012],  # Second row tests future year data entry anomaly
        "YearRemodAdd": [2005, 2008],
        "FullBath": [2, 1],
        "HalfBath": [1, 0],
        "BsmtFullBath": [1, 0],
        "BsmtHalfBath": [0, 1],
        "GrLivArea": [1500, 2000],
        "TotalBsmtSF": [800, 1000],
        "OpenPorchSF": [50, 0],
        "EnclosedPorch": [0, 20],
        "3SsnPorch": [0, 0],
        "ScreenPorch": [0, 0],
        "WoodDeckSF": [100, 0],
        "GarageArea": [400, 0],
        "PoolArea": [0, 100],
        "OverallQual": [7, 9]
    })
    
    fe = rp.FeatureEngineer()
    out_df = fe.transform(sample_df)
    
    # HouseAge checks
    assert "HouseAge" in out_df.columns
    assert out_df.loc[0, "HouseAge"] == 10
    assert out_df.loc[1, "HouseAge"] == 0  # Capped at 0 for future entry anomaly
    
    # TotalBathrooms checks (Full + 0.5*Half + BsmtFull + 0.5*BsmtHalf)
    assert "TotalBathrooms" in out_df.columns
    assert out_df.loc[0, "TotalBathrooms"] == 3.5
    
    # TotalLivingArea checks (GrLivArea + TotalBsmtSF)
    assert "TotalLivingArea" in out_df.columns
    assert out_df.loc[0, "TotalLivingArea"] == 2300

# ============================================================
# 3. Model & Pipeline Inference Tests
# ============================================================
def test_model_pipeline_loads_and_predicts():
    """Ensure serialized model pipeline loads and produces valid non-negative predictions."""
    assert config.BEST_PIPELINE_JOBLIB.exists(), f"Model pipeline not found at {config.BEST_PIPELINE_JOBLIB}"
    pipeline = joblib.load(config.BEST_PIPELINE_JOBLIB)
    
    df = utils.load_data(config.TRAIN_DATA_PATH, is_train=True)
    X = df.drop(columns=[config.TARGET_COL, config.ID_COL], errors="ignore").head(20)
    
    preds = pipeline.predict(X)
    assert len(preds) == 20
    assert not np.isnan(preds).any(), "Model produced NaN predictions."
    assert (preds > 0).all(), "Model produced non-positive predictions."

# ============================================================
# 4. Business KPI Module Tests
# ============================================================
def test_business_kpi_computation():
    """Verify business KPI calculations."""
    y_true = pd.Series([100000, 200000, 300000])
    y_pred = np.array([105000, 190000, 350000])
    dummy_df = pd.DataFrame({"YrSold": [2008, 2009, 2010]})
    
    kpis = kpi_metrics.compute_business_kpis(dummy_df, y_true, y_pred)
    
    assert "volume_and_scale" in kpis
    assert "accuracy_and_quality_sla" in kpis
    assert "investment_arbitrage_and_profit" in kpis
    assert kpis["volume_and_scale"]["total_properties_evaluated"] == 3
    assert kpis["volume_and_scale"]["estimated_appraisal_cost_savings_usd"] == 1200.0

# ============================================================
# 5. Recommendation Engine Tests
# ============================================================
def test_recommendation_decision_rules():
    """Verify deal evaluation actions."""
    engine = rec_engine.RealEstateRecommendationEngine()
    
    # 1. 20% Undervalued (Actual: $100k, Fair Value: $120k) -> STRONG BUY
    buy_deal = engine.evaluate_deal(actual_price=100000, predicted_price=120000)
    assert buy_deal["action"] == "STRONG BUY"
    assert buy_deal["risk_level"] == "Low"
    
    # 2. Fair Value (Actual: $100k, Fair Value: $102k) -> FAIR VALUE
    fair_deal = engine.evaluate_deal(actual_price=100000, predicted_price=102000)
    assert fair_deal["action"] == "FAIR VALUE"
    
    # 3. 20% Overpriced (Actual: $120k, Fair Value: $100k) -> PASS / NEGOTIATE
    pass_deal = engine.evaluate_deal(actual_price=120000, predicted_price=100000)
    assert pass_deal["action"] == "PASS / NEGOTIATE"
    assert pass_deal["risk_level"] == "High"
