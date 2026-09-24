import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import joblib

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.config as config
import src.utils as utils

class RealEstateRecommendationEngine:
    """
    Decision support and recommendation engine that translates ML regression predictions
    into commercial real estate actions (Acquisition, Counter-Offers, and Renovation ROI).
    """
    def __init__(self, pipeline: Optional[Any] = None) -> None:
        if pipeline is None:
            if not config.BEST_PIPELINE_JOBLIB.exists():
                raise FileNotFoundError(f"Model pipeline not found at {config.BEST_PIPELINE_JOBLIB}. Please run train.py first.")
            self.pipeline = joblib.load(config.BEST_PIPELINE_JOBLIB)
        else:
            self.pipeline = pipeline

    def evaluate_deal(self, actual_price: float, predicted_price: float) -> Dict[str, Any]:
        """
        Classifies a real estate transaction and provides an actionable decision recommendation.
        """
        spread = predicted_price - actual_price  # Positive = Undervalued bargain
        pct_spread = (spread / actual_price) * 100.0 if actual_price > 0 else 0.0
        
        if pct_spread >= 10.0:
            action = "STRONG BUY"
            risk_level = "Low"
            recommendation = (
                f"Undervalued by {pct_spread:.1f}% (${spread:,.0f} upside). "
                f"Acquire immediately below model fair value of ${predicted_price:,.0f}."
            )
        elif pct_spread <= -10.0:
            action = "PASS / NEGOTIATE"
            risk_level = "High"
            max_counter_offer = predicted_price * 0.98
            recommendation = (
                f"Overpriced by {abs(pct_spread):.1f}% (Overpayment risk: ${abs(spread):,.0f}). "
                f"Do not purchase at ask price. Submit counter-offer capped at ${max_counter_offer:,.0f}."
            )
        else:
            action = "FAIR VALUE"
            risk_level = "Moderate"
            recommendation = (
                f"Priced at fair market value (spread: {pct_spread:+.1f}%). "
                f"Standard mortgage underwriting and acquisition approved."
            )
            
        return {
            "action": action,
            "risk_level": risk_level,
            "actual_price": float(actual_price),
            "predicted_fair_value": float(predicted_price),
            "dollar_spread": float(spread),
            "percentage_spread": float(pct_spread),
            "recommendation": recommendation
        }

    def simulate_renovation_roi(self, property_row: pd.Series) -> List[Dict[str, Any]]:
        """
        Simulates hypothetical property renovations and estimates value appreciation.
        """
        baseline_df = pd.DataFrame([property_row.to_dict()])
        if config.TARGET_COL in baseline_df.columns:
            baseline_df = baseline_df.drop(columns=[config.TARGET_COL])
        if config.ID_COL in baseline_df.columns:
            baseline_df = baseline_df.drop(columns=[config.ID_COL])
            
        baseline_price = float(self.pipeline.predict(baseline_df)[0])
        renovation_scenarios = []
        
        # Scenario 1: Quality Upgrade (+1 OverallQual point, estimated cost: $12,000)
        curr_qual = property_row.get("OverallQual", 5)
        if curr_qual < 10:
            qual_df = baseline_df.copy()
            qual_df["OverallQual"] = min(10, curr_qual + 1)
            new_price = float(self.pipeline.predict(qual_df)[0])
            value_lift = new_price - baseline_price
            est_cost = 12000.0
            net_roi = ((value_lift - est_cost) / est_cost) * 100.0
            renovation_scenarios.append({
                "renovation": "Material & Finish Quality Upgrade (+1 OverallQual)",
                "estimated_cost": est_cost,
                "projected_value_lift": value_lift,
                "projected_new_valuation": new_price,
                "estimated_roi_pct": net_roi,
                "recommendation": "Recommended" if net_roi > 15 else "Marginal ROI"
            })
            
        # Scenario 2: Add 1 Full Bathroom (estimated cost: $15,000)
        curr_bath = property_row.get("FullBath", 1)
        bath_df = baseline_df.copy()
        bath_df["FullBath"] = curr_bath + 1
        new_price_bath = float(self.pipeline.predict(bath_df)[0])
        value_lift_bath = new_price_bath - baseline_price
        est_cost_bath = 15000.0
        net_roi_bath = ((value_lift_bath - est_cost_bath) / est_cost_bath) * 100.0
        renovation_scenarios.append({
            "renovation": "Add 1 Full Bathroom (+1 FullBath)",
            "estimated_cost": est_cost_bath,
            "projected_value_lift": value_lift_bath,
            "projected_new_valuation": new_price_bath,
            "estimated_roi_pct": net_roi_bath,
            "recommendation": "Recommended" if net_roi_bath > 15 else "Low ROI"
        })
        
        # Scenario 3: Finish Basement Living Area (+300 sq ft, estimated cost: $10,000)
        curr_bsmt = property_row.get("TotalBsmtSF", 0)
        bsmt_df = baseline_df.copy()
        bsmt_df["TotalBsmtSF"] = curr_bsmt + 300
        new_price_bsmt = float(self.pipeline.predict(bsmt_df)[0])
        value_lift_bsmt = new_price_bsmt - baseline_price
        est_cost_bsmt = 10000.0
        net_roi_bsmt = ((value_lift_bsmt - est_cost_bsmt) / est_cost_bsmt) * 100.0
        renovation_scenarios.append({
            "renovation": "Expand/Finish Basement (+300 sq ft TotalBsmtSF)",
            "estimated_cost": est_cost_bsmt,
            "projected_value_lift": value_lift_bsmt,
            "projected_new_valuation": new_price_bsmt,
            "estimated_roi_pct": net_roi_bsmt,
            "recommendation": "Highly Recommended" if net_roi_bsmt > 25 else "Recommended"
        })
        
        return renovation_scenarios

def generate_recommendations() -> pd.DataFrame:
    """
    Runs decision analysis on the validation cohort and exports structured recommendation reports.
    """
    print("=" * 60)
    print("Real Estate Decision & Recommendation Engine")
    print("=" * 60)
    
    engine = RealEstateRecommendationEngine()
    
    df = utils.load_data(config.TRAIN_DATA_PATH, is_train=True)
    X = df.drop(columns=[config.TARGET_COL, config.ID_COL], errors="ignore")
    y = df[config.TARGET_COL]
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    y_val_pred = engine.pipeline.predict(X_val)
    y_val_pred = np.clip(y_val_pred, a_min=0, a_max=None)
    
    records = []
    val_indices = y_val.index
    for idx, actual, pred in zip(val_indices, y_val.values, y_val_pred):
        deal = engine.evaluate_deal(actual_price=actual, predicted_price=pred)
        records.append({
            "PropertyIndex": int(idx),
            "ActualSalePrice": deal["actual_price"],
            "PredictedFairValue": deal["predicted_fair_value"],
            "DollarSpread": deal["dollar_spread"],
            "PercentageSpread": deal["percentage_spread"],
            "Action": deal["action"],
            "RiskLevel": deal["risk_level"],
            "DecisionRecommendation": deal["recommendation"]
        })
        
    rec_df = pd.DataFrame(records)
    config.RECOMMENDATIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rec_df.to_csv(config.RECOMMENDATIONS_CSV, index=False)
    print(f"Saved recommendations CSV to {config.RECOMMENDATIONS_CSV}")
    
    # Generate Renovation ROI example for median property
    median_idx = int(rec_df["ActualSalePrice"].sub(rec_df["ActualSalePrice"].median()).abs().idxmin())
    sample_row = df.loc[rec_df.loc[median_idx, "PropertyIndex"]]
    renovations = engine.simulate_renovation_roi(sample_row)
    
    # Generate Markdown Report
    report_content = f"""# Real Estate Decision & Recommendation Engine Report

This report operationalizes our Machine Learning model predictions into concrete decision rules for real estate acquisitions, price negotiations, and value-add renovation strategies.

---

## 1. Executive Deal Classification Summary

| Action Category | Count | Percentage | Commercial Strategy |
| :--- | :---: | :---: | :--- |
| **STRONG BUY** (Undervalued >= 10%) | **{int((rec_df['Action'] == 'STRONG BUY').sum())} homes** | **{((rec_df['Action'] == 'STRONG BUY').mean() * 100):.1f}%** | Acquire below fair value to capture instant equity / arbitrage profit |
| **FAIR VALUE** (Within ±10%) | **{int((rec_df['Action'] == 'FAIR VALUE').sum())} homes** | **{((rec_df['Action'] == 'FAIR VALUE').mean() * 100):.1f}%** | Fair market pricing; standard mortgage underwriting approved |
| **PASS / NEGOTIATE** (Overpriced >= 10%) | **{int((rec_df['Action'] == 'PASS / NEGOTIATE').sum())} homes** | **{((rec_df['Action'] == 'PASS / NEGOTIATE').mean() * 100):.1f}%** | Overpayment risk; submit discounted counter-offer |

---

## 2. Top 5 Investment Arbitrage Opportunities (Strong Buy)

The following properties represent the highest potential profit margins (Actual Price significantly below Model Fair Value):

| Property Index | Actual Price | Predicted Fair Value | Instant Equity Spread ($) | Profit Margin (%) |
| :---: | :---: | :---: | :---: | :---: |
"""
    top_buys = rec_df[rec_df["Action"] == "STRONG BUY"].sort_values(by="DollarSpread", ascending=False).head(5)
    for _, row in top_buys.iterrows():
        report_content += f"| {int(row['PropertyIndex'])} | ${row['ActualSalePrice']:,.2f} | ${row['PredictedFairValue']:,.2f} | ${row['DollarSpread']:,.2f} | +{row['PercentageSpread']:.1f}% |\n"
        
    report_content += """
---

## 3. Top 5 High-Risk Overpriced Properties (Negotiate or Pass)

The following properties require aggressive price negotiations to prevent overpayment:

| Property Index | Actual Price | Predicted Fair Value | Overpricing Spread ($) | Downside Premium (%) |
| :---: | :---: | :---: | :---: | :---: |
"""
    top_risks = rec_df[rec_df["Action"] == "PASS / NEGOTIATE"].sort_values(by="DollarSpread", ascending=True).head(5)
    for _, row in top_risks.iterrows():
        report_content += f"| {int(row['PropertyIndex'])} | ${row['ActualSalePrice']:,.2f} | ${row['PredictedFairValue']:,.2f} | ${abs(row['DollarSpread']):,.2f} | {abs(row['PercentageSpread']):.1f}% |\n"
        
    report_content += f"""
---

## 4. Renovation ROI Decision Simulator

Simulated value appreciation on a representative median property (Baseline Valuation: **${sample_row[config.TARGET_COL]:,.2f}**):

| Renovation Strategy | Estimated Cost | Projected Value Lift | Projected New Value | Net ROI (%) | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
"""
    for r in renovations:
        report_content += f"| **{r['renovation']}** | ${r['estimated_cost']:,.2f} | ${r['projected_value_lift']:,.2f} | ${r['projected_new_valuation']:,.2f} | **{r['estimated_roi_pct']:+.1f}%** | {r['recommendation']} |\n"
        
    report_content += """
---

## 5. Summary Decision Framework

```text
               Property Evaluation
                       │
                       ▼
            Model Fair Market Value
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  Spread ≥ +10%   Spread ±10%   Spread ≤ -10%
    STRONG BUY     FAIR VALUE   PASS / NEGOTIATE
   (Target Deal)  (Standard)   (Counter-Offer)
```
"""
    with open(config.RECOMMENDATION_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved recommendation report to {config.RECOMMENDATION_REPORT_MD}")
    
    return rec_df

if __name__ == "__main__":
    generate_recommendations()
