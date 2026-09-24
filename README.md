# Ames Real Estate Valuation & Decision Analytics Pipeline

An end-to-end, production-grade Machine Learning and Decision Support project that predicts residential property values in Ames, Iowa, and translates predictive models into commercial investment KPIs and actionable recommendations.

---

## 1. Problem Statement

* **Real-World Problem**: Real estate valuation is traditionally slow, subjective, and prone to appraisal bias. Buyers and institutional investors often overpay for properties or miss lucrative arbitrage opportunities due to imperfect market information.
* **Who Faces This Problem?**: Real estate investment trusts (REITs), mortgage underwriters, PropTech platforms, home buyers, and appraisers.
* **Why Does It Matter?**: Residential real estate represents the largest single asset class. Pricing errors of even 5–10% lead to tens of thousands of dollars in misallocated capital, delayed transactions, and elevated loan default risks.
* **Decisions Supported**: 
  1. Instant fair market valuation (automated appraisal).
  2. Deal acquisition classification (`STRONG BUY`, `FAIR VALUE`, `PASS / NEGOTIATE`).
  3. Renovation ROI prioritization (identifying which property upgrades yield the highest financial return).

---

## 2. Project Objectives

1. **Exploratory Data Analysis**: Identify key spatial, structural, and temporal drivers of residential property valuations.
2. **Leakage-Free Feature Engineering**: Dynamically engineer composite spatial and age metrics inside a unified Scikit-learn Pipeline.
3. **Model Benchmark & Optimization**: Compare baseline OLS, regularized Ridge, and Lasso regressors with cross-validated hyperparameter tuning.
4. **Two-Tier KPI Evaluation**: Measure both statistical ML KPIs ($R^2$, MAE, RMSE) and commercial business KPIs (Portfolio Valuation, AOV, Appraisal Cost Savings, Precision Tolerance Bands).
5. **Actionable Decision Engine**: Convert model predictions into automated investment recommendations and renovation ROI estimations.
6. **Automated Validation**: Ensure robustness with a 100% passing automated test suite (`pytest`).

---

## 3. Dataset Description

* **Dataset Name**: Ames Housing Dataset (Transactions in Ames, Iowa, 2006–2010).
* **Records**: 1,460 rows (Train) / 1,459 rows (Inference Test).
* **Features**: 79 explanatory variables (37 numerical, 43 categorical).
* **Target Variable**: `SalePrice` (Continuous, in USD).
* **Data Quality & Preprocessing**:
  - Duplicate filtration and outlier removal (filtering houses with `GrLivArea` > 4,000 sq ft and `SalePrice` < $300,000 per Dean De Cock's recommendations).
  - Median numerical imputation and mode/unknown categorical encoding.
  - Reproducible 80/20 train-validation split (`random_state=42`).

---

## 4. Technology Stack

* **Programming**: Python 3.11+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-learn (Pipelines, ColumnTransformer, Ridge, Lasso, GridSearchCV, SelectKBest)
* **Visual Analytics**: Matplotlib, Seaborn
* **Model Serialization**: Joblib
* **Automated Testing**: Pytest
* **Version Control**: Git & GitHub

---

## 5. End-to-End Analytics Pipeline

```text
Raw CSV Data (data/train.csv)
     ↓
Data Cleaning & Outlier Filtration
     ↓
Train / Validation Split (80 / 20)
     ↓
Custom Feature Engineering (HouseAge, TotalLivingArea, TotalBathrooms)
     ↓
ColumnTransformer (Median Imputation, One-Hot Encoding, StandardScaler)
     ↓
Feature Selection (SelectKBest f_regression)
     ↓
Tuned Estimator (Lasso Regularization α=50.0)
     ↓
Two-Tier Evaluation (Model KPIs & Business KPIs)
     ↓
Explainability & Diagnostic Visualizations (outputs/*.png)
     ↓
Decision & Recommendation Engine (outputs/property_recommendations.csv)
```

---

## 6. Project Architecture & File Organization

```
house-price-prediction/
│
├── data/
│   ├── train.csv                      # Historical transaction dataset
│   └── test.csv                       # Unseen inference dataset
│
├── notebooks/
│   └── EDA.ipynb                      # Exploratory Data Analysis notebook
│
├── src/
│   ├── config.py                      # Global paths, constants, and hyperparameter grids
│   ├── utils.py                       # Data loaders, metric calculations, and plotting routines
│   ├── regression_pipeline.py         # Custom FeatureEngineer & ColumnTransformer pipeline
│   ├── train.py                       # Cross-validation training orchestrator
│   ├── predict.py                     # Batch inference script for unseen properties
│   ├── kpi_metrics.py                 # Commercial & operational Business KPI engine
│   └── recommendation_engine.py       # Deal evaluation & Renovation ROI advisor
│
├── tests/
│   └── test_pipeline.py               # Pytest automated test suite (Data, Pipeline, KPIs, Rules)
│
├── models/
│   └── house_price_model.pkl          # Serialized production pipeline
│
├── outputs/                           # Analytics deliverables & visual plots
│   ├── metrics.txt                    # ML metrics report
│   ├── model_comparison.csv           # Model comparison summary table
│   ├── business_kpis.json             # Structured executive KPI data
│   ├── business_kpi_report.md         # Executive Business KPI report
│   ├── business_kpi_dashboard.png     # 4-Panel Executive visual dashboard
│   ├── property_recommendations.csv   # Scored property decisions (Buy / Pass / Negotiate)
│   ├── recommendation_report.md       # Investment & Renovation ROI strategy report
│   ├── correlation_heatmap.png        # Feature correlation map
│   ├── residuals.png                  # Prediction residual analysis
│   ├── pred_vs_actual.png             # Predicted vs. Actual scatter plot
│   ├── feature_importance.png         # Absolute coefficient importance
│   └── learning_curve.png             # Sample complexity & generalization curve
│
├── requirements.txt                   # Pinned package dependencies
└── README.md                          # Project documentation
```

---

## 7. Model Evaluation & Comparison

| Model Configuration | Polynomial Features | Feature Selection | Validation R² | Validation MAE | Validation RMSE | CV RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Base)** | No | No | 0.7049 | $18,350.91 | $40,372.14 | $26,104.82 |
| **Ridge Regression (Tuned)** | No | No | 0.9182 | $15,339.82 | $21,253.62 | $23,110.09 |
| **Lasso Regression (Tuned)** | No | No | **0.9184** | **$15,182.56** | **$21,228.34** | **$23,679.02** |
| **Linear Regression (Poly)** | Yes | k=50 | 0.8675 | $19,271.66 | $27,048.72 | $27,118.76 |
| **Ridge Regression (Poly)** | Yes | k=50 | 0.8692 | $19,246.23 | $26,880.60 | $26,807.45 |
| **Lasso Regression (Poly)** | Yes | k=50 | 0.8644 | $19,456.47 | $27,368.44 | $26,787.28 |

*Key Takeaway*: Tuned Lasso Regression ($\alpha=50.0$) achieves the best performance with an $R^2$ of **91.84%** and an MAE of **$15,182.56**.

---

## 8. Two-Tier KPI Framework

### Tier 1: Model Statistical KPIs
- **$R^2$ Score**: `0.9184` (91.84% variance explained)
- **Mean Absolute Error (MAE)**: `$15,182.56`
- **Root Mean Squared Error (RMSE)**: `$21,228.34`
- **Mean Absolute Percentage Error (MAPE)**: `9.26%`

### Tier 2: Business & Commercial KPIs
- **Valuation Scale (Orders)**: **292 properties** evaluated in holdout validation; **1,459 properties** in inference batch.
- **Portfolio Asset Volume**: **$52.96 Million** assessed in validation; **$261.28 Million** in inference batch.
- **Average Property Valuation (AOV)**: **$181,722.94**
- **Automated Appraisal Savings**: **$116,800.00** saved in holdout ($400/appraisal automated); **$583,600.00** on inference dataset.
- **Valuation Accuracy Within ±10% (SLA)**: **69.52%** of properties.
- **Valuation Accuracy Within ±20% (SLA)**: **94.18%** of properties.
- **Arbitrage Opportunities Identified**: **47 properties (16.1%)** priced $\ge 10\%$ below fair value.
- **Gross Arbitrage Profit Potential**: **$1,572,846.41** in uncaptured market equity.

---

## 9. Decision & Recommendation Layer

The recommendation engine converts predictions into actionable commercial strategies:

1. **Deal Acquisition Classifier**:
   - `STRONG BUY`: Spread $\ge +10\%$ $\to$ Undervalued asset, capture immediate equity.
   - `FAIR VALUE`: Spread within $\pm 10\%$ $\to$ Standard pricing, approved for underwriting.
   - `PASS / NEGOTIATE`: Spread $\le -10\%$ $\to$ Overpayment risk, submit discounted counter-offer.
2. **Renovation ROI Simulator**:
   - Tests physical upgrades (Quality upgrades, additional bathrooms, finished basements) and returns the projected dollar lift and net ROI percentage.

---

## 10. Explainability & Insights

Visual diagnostic plots in `outputs/`:
- **Feature Importance (`outputs/feature_importance.png`)**: Highlights top positive valuation drivers: `TotalLivingArea`, `OverallQual`, `YearBuilt`, `TotalBsmtSF`, and premium neighborhoods (`NridgHt`, `StoneBr`).
- **Residual Analysis (`outputs/residuals.png`)**: Confirms error homoscedasticity across price ranges.
- **Predicted vs. Actual (`outputs/pred_vs_actual.png`)**: Demonstrates tight clustering around the 45-degree line.

---

## 11. Testing & Validation

Run the automated test suite to verify data integrity, custom pipeline transformations, model outputs, KPIs, and decision rules:
```bash
pytest -v
```
*Result: 6/6 unit & integration tests passing.*

---

## 12. How to Run the Project

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Model Training & Diagnostics**:
   ```bash
   python src/train.py
   ```

3. **Run Inference on Unseen Properties**:
   ```bash
   python src/predict.py
   ```

4. **Compute Business KPIs & Generate Dashboard**:
   ```bash
   python src/kpi_metrics.py
   ```

5. **Run Decision & Recommendation Engine**:
   ```bash
   python src/recommendation_engine.py
   ```

6. **Run Automated Test Suite**:
   ```bash
   pytest -v
   ```
