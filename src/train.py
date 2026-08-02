import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.config as config
import src.utils as utils
import src.regression_pipeline as rp

def train_and_evaluate() -> None:
    """
    Downloads raw data, processes feature engineering and preprocessing, 
    tunes hyperparameters via GridSearchCV, compares regression models,
    outputs metric logs, plots, and saves the best production pipeline.
    """
    print("=" * 60)
    print("Ames House Price Prediction Training Workflow")
    print("=" * 60)
    
    
    utils.download_dataset()
    df = utils.load_data(config.TRAIN_DATA_PATH, is_train=True)
    
    X = df.drop(columns=[config.TARGET_COL, config.ID_COL], errors="ignore")
    y = df[config.TARGET_COL]
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    utils.plot_correlation_heatmap(df, config.OUTPUT_DIR / "correlation_heatmap.png")
    print("Saved correlation heatmap to outputs/correlation_heatmap.png")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Dataset Split: Train: {X_train.shape[0]} rows, Validation: {X_val.shape[0]} rows")
    
    experiments = [
        {
            "name": "Linear Regression (Base)",
            "model": LinearRegression(),
            "poly": False,
            "k_best": None,
            "grid": None
        },
        {
            "name": "Ridge Regression (Tuned)",
            "model": Ridge(random_state=config.RANDOM_STATE),
            "poly": False,
            "k_best": None,
            "grid": config.RIDGE_PARAM_GRID
        },
        {
            "name": "Lasso Regression (Tuned)",
            "model": Lasso(random_state=config.RANDOM_STATE, max_iter=10000),
            "poly": False,
            "k_best": None,
            "grid": config.LASSO_PARAM_GRID
        },
        {
            "name": "Linear Regression + Poly + SelectKBest",
            "model": LinearRegression(),
            "poly": True,
            "k_best": 50,  # Select top 50 features to avoid overfitting with polynomial features
            "grid": None
        },
        {
            "name": "Ridge + Poly + SelectKBest (Tuned)",
            "model": Ridge(random_state=config.RANDOM_STATE),
            "poly": True,
            "k_best": 50,
            "grid": config.RIDGE_PARAM_GRID
        },
        {
            "name": "Lasso + Poly + SelectKBest (Tuned)",
            "model": Lasso(random_state=config.RANDOM_STATE, max_iter=10000),
            "poly": True,
            "k_best": 50,
            "grid": config.LASSO_PARAM_GRID
        }
    ]
    
    results = []
    trained_pipelines = {}
    
    for exp in experiments:
        name = exp["name"]
        print(f"\nRunning experiment: {name}...")
        
        pipeline = rp.build_regression_pipeline(
            model=exp["model"],
            include_polynomials=exp["poly"],
            k_best_features=exp["k_best"]
        )
        
        if exp["grid"]:
            print(f"Tuning hyperparameters with GridSearchCV ({config.CV_FOLDS}-fold)...")
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=exp["grid"],
                scoring="neg_root_mean_squared_error",
                cv=config.CV_FOLDS,
                n_jobs=-1
            )
            try:
                grid_search.fit(X_train, y_train)
                best_pipeline = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best Hyperparameters: {best_params}")
            except Exception as e:
                print(f"Grid search failed for {name}: {e}. Falling back to default pipeline.")
                best_pipeline = pipeline
                best_pipeline.fit(X_train, y_train)
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)
            
        y_train_pred = best_pipeline.predict(X_train)
        y_val_pred = best_pipeline.predict(X_val)
        
        metrics = utils.evaluate_metrics(y_val, y_val_pred)
        
        print("Computing cross-validation scores...")
        cv_scores = cross_val_score(
            best_pipeline, X_train, y_train,
            cv=config.CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        mean_cv_rmse = -cv_scores.mean()
        
        results.append({
            "Model": name,
            "Polynomials": "Yes" if exp["poly"] else "No",
            "SelectKBest": f"k={exp['k_best']}" if exp["k_best"] else "No",
            "R2": metrics["R2"],
            "MAE": metrics["MAE"],
            "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"],
            "CV RMSE": mean_cv_rmse
        })
        
        trained_pipelines[name] = {
            "pipeline": best_pipeline,
            "y_train_pred": y_train_pred,
            "y_val_pred": y_val_pred,
            "rmse": metrics["RMSE"]
        }
        
        print(f"Validation R2: {metrics['R2']:.4f} | RMSE: ${metrics['RMSE']:.2f} | CV RMSE: ${mean_cv_rmse:.2f}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Ames House Price Prediction - Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
    best_model_idx = results_df["RMSE"].idxmin()
    best_model_name = results_df.loc[best_model_idx, "Model"]
    best_metrics = results_df.loc[best_model_idx]
    
    print(f"\n--> Best-performing model: {best_model_name}")
    print(f"    Validation RMSE: ${best_metrics['RMSE']:.2f}")
    print(f"    Validation R2: {best_metrics['R2']:.4f}")
    
    with open(config.METRICS_PATH, "a") as f:
        f.write("=" * 60 + "\n")
        f.write(f"BEST PERFORMING MODEL DETAILS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model Name: {best_model_name}\n")
        f.write(f"Polynomial Features: {best_metrics['Polynomials']}\n")
        f.write(f"Feature Selection: {best_metrics['SelectKBest']}\n")
        f.write(f"Validation R2: {best_metrics['R2']:.5f}\n")
        f.write(f"Validation MAE: ${best_metrics['MAE']:.2f}\n")
        f.write(f"Validation RMSE: ${best_metrics['RMSE']:.2f}\n")
        f.write(f"Cross-Validation RMSE: ${best_metrics['CV RMSE']:.2f}\n")
        f.write("=" * 60 + "\n")
        
    print(f"Saved evaluation metrics report to {config.METRICS_PATH}")
    
    best_pipeline_info = trained_pipelines[best_model_name]
    best_pipeline = best_pipeline_info["pipeline"]
    
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, config.MODEL_PATH)
    print(f"Saved the best trained pipeline to {config.MODEL_PATH}")
    
    print("\nGenerating diagnostic plots for the best-performing model...")
    
    utils.plot_residuals(
        y_train=y_train.values,
        y_train_pred=best_pipeline_info["y_train_pred"],
        y_val=y_val.values,
        y_val_pred=best_pipeline_info["y_val_pred"],
        save_path=config.OUTPUT_DIR / "residuals.png"
    )
    print("Saved residuals plot to outputs/residuals.png")
    
    utils.plot_pred_vs_actual(
        y_val=y_val.values,
        y_val_pred=best_pipeline_info["y_val_pred"],
        save_path=config.OUTPUT_DIR / "pred_vs_actual.png"
    )
    print("Saved prediction vs actual plot to outputs/pred_vs_actual.png")
    
    utils.plot_feature_importance(
        pipeline=best_pipeline,
        save_path=config.OUTPUT_DIR / "feature_importance.png"
    )
    print("Saved feature importance plot to outputs/feature_importance.png")
    
    utils.plot_learning_curves(
        pipeline=best_pipeline,
        X=X_train,
        y=y_train.values,
        save_path=config.OUTPUT_DIR / "learning_curve.png"
    )
    print("Saved learning curve plot to outputs/learning_curve.png")
    
    utils.plot_model_comparison(
        results_df=results_df,
        metric="RMSE",
        save_path=config.OUTPUT_DIR / "model_comparison.png"
    )
    print("Saved model comparison plot to outputs/model_comparison.png")

if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"Error in training workflow: {e}", file=sys.stderr)
        sys.exit(1)
