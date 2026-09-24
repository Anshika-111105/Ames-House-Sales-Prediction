import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import src.config as config
import src.utils as utils

def compute_business_kpis(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes commercial and operational Real Estate KPIs from validation predictions."""
    actual = np.array(y_true)
    pred = np.array(y_pred)
    
    abs_errors = np.abs(actual - pred)
    pct_errors = (abs_errors / actual) * 100.0
    dollar_spread = pred - actual  # Positive = Model thinks it's worth more (Undervalued bargain)
    
    # 1. Volume & Scale KPIs (Orders, Customers, Asset scale)
    total_properties = len(actual)
    total_portfolio_actual = float(np.sum(actual))
    total_portfolio_pred = float(np.sum(pred))
    avg_property_val = float(np.mean(pred))
    median_property_val = float(np.median(pred))
    
    # Automated appraisal savings (benchmark: $400 and 4 hours per manual appraisal)
    appraisal_cost_per_unit = 400.0
    appraisal_hours_per_unit = 4.0
    total_appraisal_savings = float(total_properties * appraisal_cost_per_unit)
    total_hours_saved = float(total_properties * appraisal_hours_per_unit)
    
    # 2. Valuation Precision & SLA KPIs (Quality / Risk)
    within_5_pct = float(np.mean(pct_errors <= 5.0) * 100.0)
    within_10_pct = float(np.mean(pct_errors <= 10.0) * 100.0)
    within_15_pct = float(np.mean(pct_errors <= 15.0) * 100.0)
    within_20_pct = float(np.mean(pct_errors <= 20.0) * 100.0)
    high_risk_miss_rate = float(np.mean(pct_errors > 20.0) * 100.0)
    mape = float(np.mean(pct_errors))
    mdape = float(np.median(pct_errors))
    
    # 3. Investment Arbitrage & Profitability KPIs (Profit & Margins)
    # Undervalued: Model Fair Value >= 10% higher than actual sale price (Bargain acquisition opportunity)
    undervalued_mask = (dollar_spread / actual) >= 0.10
    undervalued_count = int(np.sum(undervalued_mask))
    undervalued_pct = float((undervalued_count / total_properties) * 100.0)
    gross_arbitrage_profit = float(np.sum(dollar_spread[undervalued_mask])) if undervalued_count > 0 else 0.0
    avg_arbitrage_upside = float(np.mean((dollar_spread[undervalued_mask] / actual[undervalued_mask]) * 100.0)) if undervalued_count > 0 else 0.0
    
    # Overvalued: Model Fair Value <= 10% lower than actual price (Overpayment risk flagging)
    overvalued_mask = (dollar_spread / actual) <= -0.10
    overvalued_count = int(np.sum(overvalued_mask))
    overvalued_pct = float((overvalued_count / total_properties) * 100.0)
    downside_risk_exposure = float(np.sum(np.abs(dollar_spread[overvalued_mask]))) if overvalued_count > 0 else 0.0
    
    fairly_priced_count = total_properties - undervalued_count - overvalued_count
    fairly_priced_pct = float((fairly_priced_count / total_properties) * 100.0)
    
    # 4. Price Tier Segmentation
    tiers = []
    for limit, label in [(130000, "Entry-Level (<$130k)"), (250000, "Mid-Market ($130k-$250k)"), (float('inf'), "Premium/Luxury (>$250k)")]:
        if label.startswith("Entry"):
            mask = actual < 130000
        elif label.startswith("Mid"):
            mask = (actual >= 130000) & (actual <= 250000)
        else:
            mask = actual > 250000
            
        count = int(np.sum(mask))
        if count > 0:
            tier_mape = float(np.mean(pct_errors[mask]))
            tier_avg_price = float(np.mean(actual[mask]))
            tiers.append({
                "tier": label,
                "count": count,
                "pct_of_total": float((count / total_properties) * 100.0),
                "avg_price": tier_avg_price,
                "mape": tier_mape
            })
            
    # 5. Temporal Market Trends & YoY Growth (if YrSold exists in df)
    yearly_trends = []
    if "YrSold" in df.columns:
        df_eval = df.copy()
        df_eval["ActualPrice"] = actual
        df_eval["PredPrice"] = pred
        
        grouped = df_eval.groupby("YrSold")
        prev_vol = None
        prev_avg_price = None
        
        for yr in sorted(df_eval["YrSold"].dropna().unique()):
            sub = df_eval[df_eval["YrSold"] == yr]
            vol = len(sub)
            tot_val = float(sub["ActualPrice"].sum())
            avg_p = float(sub["ActualPrice"].mean())
            
            vol_growth = None if prev_vol is None else float(((vol - prev_vol) / prev_vol) * 100.0)
            price_growth = None if prev_avg_price is None else float(((avg_p - prev_avg_price) / prev_avg_price) * 100.0)
            
            yearly_trends.append({
                "year": int(yr),
                "transactions": vol,
                "total_volume_usd": tot_val,
                "avg_price_usd": avg_p,
                "volume_growth_pct": vol_growth,
                "price_growth_pct": price_growth
            })
            prev_vol = vol
            prev_avg_price = avg_p

    return {
        "volume_and_scale": {
            "total_properties_evaluated": total_properties,
            "total_portfolio_actual_usd": total_portfolio_actual,
            "total_portfolio_predicted_usd": total_portfolio_pred,
            "average_property_valuation_aov": avg_property_val,
            "median_property_valuation": median_property_val,
            "estimated_appraisal_cost_savings_usd": total_appraisal_savings,
            "estimated_hours_saved": total_hours_saved
        },
        "accuracy_and_quality_sla": {
            "within_5_percent_precision": within_5_pct,
            "within_10_percent_precision": within_10_pct,
            "within_15_percent_precision": within_15_pct,
            "within_20_percent_precision": within_20_pct,
            "high_risk_miss_rate": high_risk_miss_rate,
            "mean_absolute_percentage_error_mape": mape,
            "median_absolute_percentage_error_mdape": mdape
        },
        "investment_arbitrage_and_profit": {
            "undervalued_opportunities_count": undervalued_count,
            "undervalued_opportunities_pct": undervalued_pct,
            "gross_arbitrage_profit_potential_usd": gross_arbitrage_profit,
            "avg_arbitrage_upside_pct": avg_arbitrage_upside,
            "overvalued_flagged_count": overvalued_count,
            "overvalued_flagged_pct": overvalued_pct,
            "downside_risk_exposure_usd": downside_risk_exposure,
            "fairly_priced_count": fairly_priced_count,
            "fairly_priced_pct": fairly_priced_pct
        },
        "price_tier_breakdown": tiers,
        "yearly_market_growth": yearly_trends
    }

def generate_kpi_dashboard_plot(kpis: Dict[str, Any], save_path: Path) -> None:
    """
    Renders a high-resolution 4-panel executive KPI dashboard image.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0f172a")  # Dark sleek slate background
    
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            
    # Panel 1: Executive KPI Cards
    ax1 = axes[0, 0]
    ax1.axis("off")
    ax1.set_title("Executive Business KPIs & Scale", color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
    
    vol = kpis["volume_and_scale"]
    arb = kpis["investment_arbitrage_and_profit"]
    
    cards = [
        ("Total Properties Valued", f"{vol['total_properties_evaluated']:,} units", "#38bdf8"),
        ("Average Property Valuation (AOV)", f"${vol['average_property_valuation_aov']:,.0f}", "#4ade80"),
        ("Total Portfolio Value Assessed", f"${vol['total_portfolio_actual_usd']/1e6:.1f} Million", "#a78bfa"),
        ("Appraisal Cost Savings", f"${vol['estimated_appraisal_cost_savings_usd']:,.0f}", "#facc15"),
        ("Gross Arbitrage Profit Potential", f"${arb['gross_arbitrage_profit_potential_usd']:,.0f}", "#34d399"),
        ("Arbitrage Opportunities Identified", f"{arb['undervalued_opportunities_count']} homes ({arb['undervalued_opportunities_pct']:.1f}%)", "#f472b6")
    ]
    
    y_positions = [0.82, 0.66, 0.50, 0.34, 0.18, 0.02]
    for i, (label, val, color) in enumerate(cards):
        y = y_positions[i]
        rect = mpatches.FancyBboxPatch((0.02, y), 0.96, 0.13, boxstyle="round,pad=0.02",
                                      facecolor="#0f172a", edgecolor=color, linewidth=1.5,
                                      transform=ax1.transAxes)
        ax1.add_patch(rect)
        ax1.text(0.06, y + 0.07, label, transform=ax1.transAxes, color="#94a3b8", fontsize=10, fontweight="medium")
        ax1.text(0.06, y + 0.025, val, transform=ax1.transAxes, color="#f8fafc", fontsize=12, fontweight="bold")
        
    # Panel 2: Valuation Precision Tolerance SLA
    ax2 = axes[0, 1]
    sla = kpis["accuracy_and_quality_sla"]
    bands = ["Within ±5%", "Within ±10%", "Within ±15%", "Within ±20%", "Miss Rate (>20%)"]
    vals = [
        sla["within_5_percent_precision"],
        sla["within_10_percent_precision"],
        sla["within_15_percent_precision"],
        sla["within_20_percent_precision"],
        sla["high_risk_miss_rate"]
    ]
    colors = ["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444"]
    
    bars = ax2.barh(bands, vals, color=colors, height=0.55, edgecolor="#0f172a", linewidth=1.2)
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Percentage of Portfolio (%)", color="#94a3b8", fontsize=11)
    ax2.set_title("Valuation Accuracy & Quality Bands", color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
    ax2.grid(axis="x", linestyle="--", alpha=0.3, color="#64748b")
    
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                 va="center", ha="left", color="#f8fafc", fontsize=10, fontweight="bold")
        
    # Panel 3: Investment Arbitrage & Risk Distribution (Donut Chart)
    ax3 = axes[1, 0]
    labels = ["Fair Value\n(Within ±10%)", "Undervalued Bargains\n(≥10% upside)", "Overvalued Risks\n(≥10% premium)"]
    sizes = [arb["fairly_priced_pct"], arb["undervalued_opportunities_pct"], arb["overvalued_flagged_pct"]]
    pie_colors = ["#3b82f6", "#10b981", "#ef4444"]
    
    wedges, texts, autotexts = ax3.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=140,
        colors=pie_colors, textprops=dict(color="#cbd5e1", fontsize=10),
        wedgeprops=dict(width=0.45, edgecolor="#1e293b", linewidth=2),
        pctdistance=0.75
    )
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontweight("bold")
    ax3.set_title("Investment Portfolio Arbitrage & Risk", color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
    
    # Panel 4: Market Dynamics & Year-over-Year (YoY) Trends
    ax4 = axes[1, 1]
    years_data = kpis["yearly_market_growth"]
    if years_data:
        yrs = [d["year"] for d in years_data]
        vols = [d["transactions"] for d in years_data]
        avg_prices = [d["avg_price_usd"] / 1000.0 for d in years_data]  # in $k
        
        ax4_twin = ax4.twinx()
        ax4_twin.tick_params(colors="#cbd5e1", labelsize=10)
        ax4_twin.spines["right"].set_color("#334155")
        ax4_twin.spines["left"].set_color("#334155")
        
        b = ax4.bar([y - 0.15 for y in yrs], vols, width=0.3, color="#38bdf8", alpha=0.85, label="Transactions (Volume)")
        l = ax4_twin.plot([y + 0.15 for y in yrs], avg_prices, color="#f59e0b", marker="o", linewidth=2.5, markersize=7, label="Avg Price ($k)")
        
        ax4.set_xlabel("Transaction Year", color="#94a3b8", fontsize=11)
        ax4.set_ylabel("Property Volume (Units)", color="#38bdf8", fontsize=11)
        ax4_twin.set_ylabel("Average Sale Price ($k)", color="#f59e0b", fontsize=11)
        ax4.set_title("Market Volume & Average Price Trends", color="#f8fafc", fontsize=14, fontweight="bold", pad=15)
        ax4.set_xticks(yrs)
        ax4.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        
        # Combined Legend
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1")
    else:
        ax4.text(0.5, 0.5, "Temporal data not available", color="#94a3b8", ha="center", va="center")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()

def generate_markdown_kpi_report(kpis: Dict[str, Any], save_path: Path) -> None:
    """
    Exports a structured business KPI report in GitHub Markdown format.
    """
    vol = kpis["volume_and_scale"]
    sla = kpis["accuracy_and_quality_sla"]
    arb = kpis["investment_arbitrage_and_profit"]
    tiers = kpis["price_tier_breakdown"]
    growth = kpis["yearly_market_growth"]
    
    content = f"""# Executive Business KPI Report: Ames Real Estate Valuation Pipeline

This report translates machine learning regression performance on the Ames Housing Dataset into commercial, operational, and investment Key Performance Indicators (KPIs).

---

## 1. Volume, Scale & Operational KPIs (Analogous to Orders & Revenue)

| Business KPI | Value | Description |
| :--- | :--- | :--- |
| **Total Properties Valued** | **{vol['total_properties_evaluated']:,} units** | Total residential properties evaluated in validation pipeline (*Volume / Order analog*) |
| **Total Portfolio Value (Actual)** | **${vol['total_portfolio_actual_usd']:,.2f}** | Cumulative market asset volume assessed (*Revenue / Asset Scale*) |
| **Total Portfolio Value (Predicted)** | **${vol['total_portfolio_predicted_usd']:,.2f}** | Cumulative model-estimated asset volume |
| **Average Property Valuation (AOV)** | **${vol['average_property_valuation_aov']:,.2f}** | Mean estimated sale price (*Average Order Value analog*) |
| **Median Property Valuation** | **${vol['median_property_valuation']:,.2f}** | Median property transaction value |
| **Automated Appraisal Cost Savings** | **${vol['estimated_appraisal_cost_savings_usd']:,.2f}** | Operational cost savings ($400 industry standard appraisal fee saved per property) |
| **Appraisal Hours Saved** | **{vol['estimated_hours_saved']:,.1f} hours** | Manual inspection and underwriting time eliminated (4 hours/appraisal) |

---

## 2. Valuation Precision & Quality SLA KPIs

| Precision Band | Accuracy Rate (%) | Industry Target | Commercial Meaning |
| :--- | :---: | :---: | :--- |
| **Within ±5% of Actual Price** | **{sla['within_5_percent_precision']:.2f}%** | ≥ 40% | High-precision valuation (direct auto-offer qualified) |
| **Within ±10% of Actual Price** | **{sla['within_10_percent_precision']:.2f}%** | ≥ 70% | Acceptable commercial tolerance for automated valuation models (AVMs) |
| **Within ±15% of Actual Price** | **{sla['within_15_percent_precision']:.2f}%** | ≥ 85% | Broad market pricing corridor |
| **Within ±20% of Actual Price** | **{sla['within_20_percent_precision']:.2f}%** | ≥ 92% | Safe underwriting threshold |
| **High-Risk Valuation Miss Rate (>20%)** | **{sla['high_risk_miss_rate']:.2f}%** | < 10% | Properties flagged for required human manual appraisal |
| **Mean Absolute Percentage Error (MAPE)** | **{sla['mean_absolute_percentage_error_mape']:.2f}%** | < 10% | Average relative percentage deviation per house |
| **Median Absolute Percentage Error (MdAPE)** | **{sla['median_absolute_percentage_error_mdape']:.2f}%** | < 8% | Robust median relative percentage deviation |

---

## 3. Investment Arbitrage & Profitability KPIs (Analogous to Profit & Margins)

Our valuation model identifies pricing discrepancies between actual market transactions and estimated fair values:

| Investment KPI | Value | Strategic Impact |
| :--- | :--- | :--- |
| **Undervalued "Bargain" Opportunities** | **{arb['undervalued_opportunities_count']} homes ({arb['undervalued_opportunities_pct']:.1f}%)** | Properties priced ≥10% below model fair value (*Buy Candidates*) |
| **Gross Arbitrage Profit Potential** | **${arb['gross_arbitrage_profit_potential_usd']:,.2f}** | Total dollar spread available from acquiring undervalued properties |
| **Average Profit Margin per Arbitrage Deal** | **{arb['avg_arbitrage_upside_pct']:.2f}%** | Average gross upside percentage on identified bargain properties |
| **Overvalued Properties Flagged** | **{arb['overvalued_flagged_count']} homes ({arb['overvalued_flagged_pct']:.1f}%)** | Properties priced ≥10% above model fair value (*Capital Protection / Sell Flags*) |
| **Downside Risk Exposure Mitigated** | **${arb['downside_risk_exposure_usd']:,.2f}** | Total overpayment capital prevented by flagging overvalued assets |

---

## 4. Price Tier Breakdown & Accuracy

| Market Tier | Count | Share (%) | Average Price | Model MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for t in tiers:
        content += f"| **{t['tier']}** | {t['count']} | {t['pct_of_total']:.1f}% | ${t['avg_price']:,.2f} | {t['mape']:.2f}% |\n"
        
    content += """
---

## 5. Temporal Trends & Year-over-Year (YoY) Growth

| Year | Transactions (Units) | YoY Volume Growth | Total Market Volume ($) | Average Sale Price ($) | YoY Price Growth |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for g in growth:
        vol_growth_str = f"{g['volume_growth_pct']:+.1f}%" if g['volume_growth_pct'] is not None else "Baseline"
        price_growth_str = f"{g['price_growth_pct']:+.1f}%" if g['price_growth_pct'] is not None else "Baseline"
        content += f"| **{g['year']}** | {g['transactions']} | {vol_growth_str} | ${g['total_volume_usd']:,.2f} | ${g['avg_price_usd']:,.2f} | {price_growth_str} |\n"
        
    content += f"""
---

## 6. Executive KPI Summary Visualizations

The visual dashboard has been generated and saved to:
`{config.BUSINESS_KPI_DASHBOARD_PNG.name}`
"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)

def run_kpi_analysis() -> Dict[str, Any]:
    """
    Loads data and the best trained pipeline, generates validation predictions,
    computes business KPIs, and exports JSON, Markdown, and Visual Dashboard artifacts.
    """
    print("=" * 60)
    print("Ames House Price Prediction - Business KPI Framework")
    print("=" * 60)
    
    if not config.BEST_PIPELINE_JOBLIB.exists():
        raise FileNotFoundError(f"Trained pipeline not found at {config.BEST_PIPELINE_JOBLIB}. Please run train.py first.")
        
    print(f"Loading best pipeline from {config.BEST_PIPELINE_JOBLIB}...")
    pipeline = joblib.load(config.BEST_PIPELINE_JOBLIB)
    
    print(f"Loading dataset from {config.TRAIN_DATA_PATH}...")
    df = utils.load_data(config.TRAIN_DATA_PATH, is_train=True)
    
    X = df.drop(columns=[config.TARGET_COL, config.ID_COL], errors="ignore")
    y = df[config.TARGET_COL]
    
    # Use exact same split to evaluate validation set KPIs
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    val_indices = y_val.index
    df_val = df.loc[val_indices].copy()
    
    print("Running inference on validation cohort for KPI computation...")
    y_val_pred = pipeline.predict(X_val)
    y_val_pred = np.clip(y_val_pred, a_min=0, a_max=None)
    
    print("Computing business and operational KPIs...")
    kpi_results = compute_business_kpis(df=df_val, y_true=y_val, y_pred=y_val_pred)
    
    # 1. Export JSON
    config.BUSINESS_KPIS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BUSINESS_KPIS_JSON, "w", encoding="utf-8") as f:
        json.dump(kpi_results, f, indent=4)
    print(f"Exported KPI JSON data to {config.BUSINESS_KPIS_JSON}")
    
    # 2. Export Markdown Report
    generate_markdown_kpi_report(kpi_results, config.BUSINESS_KPI_REPORT_MD)
    print(f"Exported Business KPI Report to {config.BUSINESS_KPI_REPORT_MD}")
    
    # 3. Export Visual Dashboard
    generate_kpi_dashboard_plot(kpi_results, config.BUSINESS_KPI_DASHBOARD_PNG)
    print(f"Exported Executive KPI Dashboard plot to {config.BUSINESS_KPI_DASHBOARD_PNG}")
    
    print("=" * 60)
    print("BUSINESS KPI SUMMARY")
    print("=" * 60)
    vol = kpi_results["volume_and_scale"]
    sla = kpi_results["accuracy_and_quality_sla"]
    arb = kpi_results["investment_arbitrage_and_profit"]
    
    print(f"1. Total Properties Evaluated (Orders):     {vol['total_properties_evaluated']:,} units")
    print(f"2. Total Asset Valuation (Revenue Scale):    ${vol['total_portfolio_actual_usd']:,.2f}")
    print(f"3. Average Property Valuation (AOV):         ${vol['average_property_valuation_aov']:,.2f}")
    print(f"4. Automated Appraisal Cost Savings:        ${vol['estimated_appraisal_cost_savings_usd']:,.2f}")
    print(f"5. Precision Band (Within ±10% of Price):    {sla['within_10_percent_precision']:.2f}%")
    print(f"6. Mean Absolute Percentage Error (MAPE):    {sla['mean_absolute_percentage_error_mape']:.2f}%")
    print(f"7. Undervalued Arbitrage Opportunities:      {arb['undervalued_opportunities_count']} homes ({arb['undervalued_opportunities_pct']:.1f}%)")
    print(f"8. Gross Arbitrage Profit Potential:         ${arb['gross_arbitrage_profit_potential_usd']:,.2f}")
    print("=" * 60)
    
    return kpi_results

if __name__ == "__main__":
    run_kpi_analysis()
