import os
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve

import src.config as config

def download_dataset() -> None:
    """
    Downloads the train.csv and test.csv datasets from public mirror URLs
    if they are not already present in the data directory.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, url, path in [
        ("train.csv", config.TRAIN_URL, config.TRAIN_DATA_PATH),
        ("test.csv", config.TEST_URL, config.TEST_DATA_PATH)
    ]:
        if not path.exists():
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, path)
                print(f"Successfully downloaded {filename} to {path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                raise e
        else:
            print(f"{filename} already exists at {path}")

def load_data(path: Path, is_train: bool = True) -> pd.DataFrame:
    """
    Loads the dataset from the specified path. If loading training data,
    performs initial cleanup, including removing duplicates and extreme outliers.
    
    Args:
        path (Path): Path to the CSV file.
        is_train (bool): True if training data, False if test data.
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}. Please download it first.")
        
    df = pd.read_csv(path)
    
    if is_train:
        # Check for duplicates (excluding the ID column)
        feature_cols = [col for col in df.columns if col != config.ID_COL]
        duplicates_count = df.duplicated(subset=feature_cols).sum()
        if duplicates_count > 0:
            print(f"Removing {duplicates_count} duplicate rows in training set...")
            df = df.drop_duplicates(subset=feature_cols)
            
        # Outlier detection based on author recommendations (Ames Housing: GrLivArea > 4000)
        # Author Dean De Cock recommends removing houses with GrLivArea > 4000 sq ft as outliers.
        if "GrLivArea" in df.columns and config.TARGET_COL in df.columns:
            outliers = df[(df["GrLivArea"] > 4000) & (df[config.TARGET_COL] < 300000)]
            if not outliers.empty:
                print(f"Removing {len(outliers)} extreme outliers (GrLivArea > 4000 and SalePrice < $300,000)...")
                df = df.drop(outliers.index)
                
    return df

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes key regression metrics: R2, MAE, MSE, RMSE.
    
    Args:
        y_true (np.ndarray): True targets.
        y_pred (np.ndarray): Predicted values.
        
    Returns:
        Dict[str, float]: Dictionary of metric names and their values.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        "R2": float(r2),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse)
    }

def get_pipeline_feature_names(pipeline: Any) -> List[str]:
    """
    Extracts the feature names output by a pipeline's preprocessing (ColumnTransformer) step.
    
    Args:
        pipeline (Any): Fitted sklearn pipeline.
        
    Returns:
        List[str]: List of feature names.
    """
    try:
        # Check if the pipeline contains a preprocessing step
        preprocessor = pipeline.named_steps.get("preprocessing")
        if preprocessor is None:
            # Fallback if preprocessing isn't direct
            return []
        
        # Get the feature names out of the ColumnTransformer
        feature_names = list(preprocessor.get_feature_names_out())
        # Clean up the output names (e.g., 'num__GrLivArea' -> 'GrLivArea')
        cleaned_names = []
        for name in feature_names:
            if name.startswith("num__"):
                cleaned_names.append(name[5:])
            elif name.startswith("cat__"):
                cleaned_names.append(name[5:])
            else:
                cleaned_names.append(name)
        return cleaned_names
    except Exception as e:
        print(f"Could not retrieve feature names from pipeline: {e}")
        return []

def plot_residuals(y_train: np.ndarray, y_train_pred: np.ndarray, 
                   y_val: np.ndarray, y_val_pred: np.ndarray, 
                   save_path: Path) -> None:
    """
    Generates and saves a residual plot comparing train and validation errors.
    """
    plt.figure(figsize=(10, 6))
    
    train_residuals = y_train - y_train_pred
    val_residuals = y_val - y_val_pred
    
    plt.scatter(y_train_pred, train_residuals, alpha=0.5, label="Train Residuals", color="#3498db")
    plt.scatter(y_val_pred, val_residuals, alpha=0.5, label="Validation Residuals", color="#e74c3c")
    
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Sale Price ($)")
    plt.ylabel("Residuals ($)")
    plt.title("Residual Analysis Plot")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pred_vs_actual(y_val: np.ndarray, y_val_pred: np.ndarray, save_path: Path) -> None:
    """
    Generates and saves a scatter plot of predicted vs actual house prices.
    """
    plt.figure(figsize=(8, 8))
    
    plt.scatter(y_val, y_val_pred, alpha=0.6, color="#2ecc71")
    
    # Perfect diagonal prediction line
    min_val = min(y_val.min(), y_val_pred.min())
    max_val = max(y_val.max(), y_val_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
    
    plt.xlabel("Actual Sale Price ($)")
    plt.ylabel("Predicted Sale Price ($)")
    plt.title("Predicted vs. Actual Sale Price")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(pipeline: Any, save_path: Path) -> None:
    """
    Generates and saves a bar plot representing feature importance or coefficient weights.
    Supports linear regression, Ridge, Lasso, etc.
    """
    try:
        model = pipeline.named_steps["model"]
        feature_selection = pipeline.named_steps.get("feature_selection")
        
        # Get raw feature names after preprocessing
        raw_features = get_pipeline_feature_names(pipeline)
        
        if not raw_features:
            print("Skipping feature importance plot due to missing feature names.")
            return
            
        # Handle feature selection masking if active
        if feature_selection and hasattr(feature_selection, "get_support"):
            support = feature_selection.get_support()
            features = [f for f, s in zip(raw_features, support) if s]
        else:
            features = raw_features
            
        # Retrieve coefficients or importances
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            print("Model does not support coefficient or feature importance plotting.")
            return
            
        # Align features and weights
        if len(features) != len(importances):
            # If lengths mismatch (e.g. polynomial features expanded features), we label generically
            features = [f"Feature {i}" for i in range(len(importances))]
            
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=importance_df, 
            y="Feature", 
            x="Importance", 
            palette="viridis",
            hue="Feature",
            legend=False
        )
        plt.title("Top 20 Feature Importance (Absolute Coefficients)")
        plt.xlabel("Importance (Absolute Weight)")
        plt.ylabel("Features")
        plt.grid(True, axis="x", linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error plotting feature importance: {e}")

def plot_learning_curves(pipeline: Any, X: pd.DataFrame, y: np.ndarray, save_path: Path) -> None:
    """
    Generates and saves a learning curve plot to check for high bias or variance.
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate learning curve values using RMSE
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, 
        cv=config.CV_FOLDS, 
        scoring="neg_root_mean_squared_error",
        n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=config.RANDOM_STATE
    )
    
    # Convert back to positive RMSE
    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)
    
    plt.plot(train_sizes, train_rmse, "o-", color="#e74c3c", label="Training RMSE")
    plt.plot(train_sizes, val_rmse, "o-", color="#2ecc71", label="Validation RMSE")
    
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE ($)")
    plt.title("Learning Curves (RMSE)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_model_comparison(results_df: pd.DataFrame, metric: str, save_path: Path) -> None:
    """
    Generates and saves a model comparison chart for a specific metric.
    """
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=results_df, 
        x="Model", 
        y=metric, 
        palette="Blues_d",
        hue="Model",
        legend=False
    )
    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """
    Generates and saves a correlation heatmap for top numerical variables related to SalePrice.
    """
    plt.figure(figsize=(12, 10))
    
    # Identify numerical columns only
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # Compute correlation with SalePrice and find top 15 features
    correlations = df[num_cols].corr()[config.TARGET_COL].abs().sort_values(ascending=False)
    top_features = correlations.index[:15]
    
    corr_matrix = df[top_features].corr()
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        square=True, 
        linewidths=0.5,
        cbar_kws={"shrink": .8}
    )
    plt.title("Correlation Heatmap (Top 15 Features vs SalePrice)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
