import os
import sys
from pathlib import Path
import pandas as pd
import joblib

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.config as config
import src.utils as utils

def make_predictions() -> None:
    """
    Loads the saved model pipeline, reads unseen test data, 
    generates house price predictions, and exports predictions.csv.
    """
    print("=" * 60)
    print("House Price Prediction - Inference Pipeline")
    print("=" * 60)
    
    if not config.MODEL_PATH.exists():
        print(f"Error: Saved model pipeline not found at {config.MODEL_PATH}.", file=sys.stderr)
        print("Please run train.py first to train and save the model.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading trained model pipeline from {config.MODEL_PATH}...")
    try:
        pipeline = joblib.load(config.MODEL_PATH)
        print("Successfully loaded model pipeline.")
    except Exception as e:
        print(f"Error loading model pipeline: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not config.TEST_DATA_PATH.exists():
        print(f"Downloading dataset to get test.csv...")
        utils.download_dataset()
        
    print(f"Loading test data from {config.TEST_DATA_PATH}...")
    try:
        df_test = utils.load_data(config.TEST_DATA_PATH, is_train=False)
        print(f"Loaded test data: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    except Exception as e:
        print(f"Error loading test data: {e}", file=sys.stderr)
        sys.exit(1)
        
    if config.ID_COL not in df_test.columns:
        raise ValueError(f"Required identifier column '{config.ID_COL}' missing from test dataset.")
        
    X_test = df_test.drop(columns=[config.ID_COL], errors="ignore")
    
    print("Generating predictions on test set...")
    try:
        # Pipeline contains custom FeatureEngineer -> ColumnTransformer -> Estimator
        predictions = pipeline.predict(X_test)
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        print("\nChecking if test columns match training columns shape...", file=sys.stderr)
        sys.exit(1)
        
    # Handle negative predictions (if any regression model extrapolates below zero)
    # Houses should have a positive price.
    predictions = pd.Series(predictions).clip(lower=0)
    
    output_df = pd.DataFrame({
        config.ID_COL: df_test[config.ID_COL],
        config.TARGET_COL: predictions
    })
    
    config.PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Predictions successfully written to {config.PREDICTIONS_PATH}")
    
    print("\n" + "-" * 40)
    print("Prediction Stats Summary:")
    print("-" * 40)
    print(f"Total Properties Predicted: {len(output_df)}")
    print(f"Average Predicted Price:   ${output_df[config.TARGET_COL].mean():,.2f}")
    print(f"Min Predicted Price:       ${output_df[config.TARGET_COL].min():,.2f}")
    print(f"Max Predicted Price:       ${output_df[config.TARGET_COL].max():,.2f}")
    print("-" * 40)

if __name__ == "__main__":
    make_predictions()
