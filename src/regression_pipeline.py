import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from typing import List, Optional, Any

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A custom Scikit-learn transformer that performs feature engineering
    on the Ames Housing dataset. It creates meaningful domain-specific
    features and handles impossible/negative values appropriately.
    """
    def __init__(self) -> None:
        pass
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        def get_col(col_name: str, default_val: float = 0.0) -> pd.Series:
            if col_name in X_out.columns:
                return X_out[col_name].fillna(default_val)
            return pd.Series(default_val, index=X_out.index)
            
        yr_sold = get_col("YrSold", 2010.0)
        yr_built = get_col("YearBuilt", 1970.0)
        yr_remod = get_col("YearRemodAdd", 1970.0)
        
        # Calculate Age and handle cases where YrSold < YearBuilt/YearRemod due to errors
        house_age = yr_sold - yr_built
        X_out["HouseAge"] = np.maximum(0, house_age)
        
        remod_age = yr_sold - yr_remod
        X_out["RemodelAge"] = np.maximum(0, remod_age)
        X_out["AgeSinceRemodel"] = np.maximum(0, remod_age)  # Adding AgeSinceRemodel as requested
        
        if "GarageYrBlt" in X_out.columns:
            # If garage year built is missing (e.g. no garage), set age equal to HouseAge
            garage_yr = X_out["GarageYrBlt"]
            garage_age = yr_sold - garage_yr
            # Replace missing garage ages with HouseAge
            garage_age = garage_age.fillna(X_out["HouseAge"])
            # Handle impossible future garage years or data entry errors
            X_out["GarageAge"] = np.maximum(0, garage_age)
        else:
            X_out["GarageAge"] = X_out["HouseAge"]
            
        full_bath = get_col("FullBath")
        half_bath = get_col("HalfBath")
        bsmt_full_bath = get_col("BsmtFullBath")
        bsmt_half_bath = get_col("BsmtHalfBath")
        X_out["TotalBathrooms"] = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath
        
        open_porch = get_col("OpenPorchSF")
        enclosed_porch = get_col("EnclosedPorch")
        three_ssn_porch = get_col("3SsnPorch")
        screen_porch = get_col("ScreenPorch")
        wood_deck = get_col("WoodDeckSF")
        X_out["TotalPorchArea"] = open_porch + enclosed_porch + three_ssn_porch + screen_porch + wood_deck
        
        gr_liv_area = get_col("GrLivArea")
        total_bsmt_sf = get_col("TotalBsmtSF")
        X_out["TotalLivingArea"] = gr_liv_area + total_bsmt_sf
        
        garage_area = get_col("GarageArea")
        X_out["HasGarage"] = (garage_area > 0).astype(int)
        
        X_out["HasBasement"] = (total_bsmt_sf > 0).astype(int)
        
        pool_area = get_col("PoolArea")
        X_out["HasPool"] = (pool_area > 0).astype(int)
        
        overall_qual = get_col("OverallQual", 5.0)
        X_out["LuxuryHome"] = ((gr_liv_area > 3000) & (overall_qual > 8)).astype(int)
        
        return X_out

def build_preprocessing_pipeline(include_polynomials: bool = False) -> ColumnTransformer:
    """
    Creates a ColumnTransformer to apply separate preprocessing pipelines to
    numerical and categorical columns. Automatically detects feature types.
    
    Args:
        include_polynomials (bool): Whether to generate polynomial features for numerical cols.
        
    Returns:
        ColumnTransformer: Preprocessor ready for use in a Pipeline.
    """
    # 1. Numerical Pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if include_polynomials:
        # Include polynomial interactions but keep degree=2 to avoid combinatorial explosion
        num_steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    num_steps.append(("scaler", StandardScaler()))
    
    num_pipeline = Pipeline(num_steps)
    
    # 2. Categorical Pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # 3. Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipeline, make_column_selector(dtype_exclude=np.number))
        ],
        remainder="drop"  # Drop other columns (like ID)
    )
    
    return preprocessor

def build_regression_pipeline(
    model: Any,
    include_polynomials: bool = False,
    k_best_features: Optional[int] = None
) -> Pipeline:
    """
    Constructs a complete end-to-end Machine Learning Pipeline.
    
    Pipeline Steps:
    1. Feature Engineering (Domain specific)
    2. Data Preprocessing (Imputation, Encoding, Scaling, optional Polynomials)
    3. Feature Selection (Optional SelectKBest)
    4. Estimator (Regression Model)
    """
    steps = [
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", build_preprocessing_pipeline(include_polynomials=include_polynomials))
    ]
    
    if k_best_features is not None:
        steps.append(("feature_selection", SelectKBest(score_func=f_regression, k=k_best_features)))
        
    steps.append(("model", model))
    
    return Pipeline(steps)
