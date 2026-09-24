# Executive Business KPI Report: Ames Real Estate Valuation Pipeline

This report translates machine learning regression performance on the Ames Housing Dataset into commercial, operational, and investment Key Performance Indicators (KPIs).

---

## 1. Volume, Scale & Operational KPIs (Analogous to Orders & Revenue)

| Business KPI | Value | Description |
| :--- | :--- | :--- |
| **Total Properties Valued** | **292 units** | Total residential properties evaluated in validation pipeline (*Volume / Order analog*) |
| **Total Portfolio Value (Actual)** | **$52,959,880.00** | Cumulative market asset volume assessed (*Revenue / Asset Scale*) |
| **Total Portfolio Value (Predicted)** | **$53,063,097.05** | Cumulative model-estimated asset volume |
| **Average Property Valuation (AOV)** | **$181,722.94** | Mean estimated sale price (*Average Order Value analog*) |
| **Median Property Valuation** | **$170,161.68** | Median property transaction value |
| **Automated Appraisal Cost Savings** | **$116,800.00** | Operational cost savings ($400 industry standard appraisal fee saved per property) |
| **Appraisal Hours Saved** | **1,168.0 hours** | Manual inspection and underwriting time eliminated (4 hours/appraisal) |

---

## 2. Valuation Precision & Quality SLA KPIs

| Precision Band | Accuracy Rate (%) | Industry Target | Commercial Meaning |
| :--- | :---: | :---: | :--- |
| **Within ±5% of Actual Price** | **42.47%** | ≥ 40% | High-precision valuation (direct auto-offer qualified) |
| **Within ±10% of Actual Price** | **69.52%** | ≥ 70% | Acceptable commercial tolerance for automated valuation models (AVMs) |
| **Within ±15% of Actual Price** | **82.88%** | ≥ 85% | Broad market pricing corridor |
| **Within ±20% of Actual Price** | **89.38%** | ≥ 92% | Safe underwriting threshold |
| **High-Risk Valuation Miss Rate (>20%)** | **10.62%** | < 10% | Properties flagged for required human manual appraisal |
| **Mean Absolute Percentage Error (MAPE)** | **9.26%** | < 10% | Average relative percentage deviation per house |
| **Median Absolute Percentage Error (MdAPE)** | **6.22%** | < 8% | Robust median relative percentage deviation |

---

## 3. Investment Arbitrage & Profitability KPIs (Analogous to Profit & Margins)

Our valuation model identifies pricing discrepancies between actual market transactions and estimated fair values:

| Investment KPI | Value | Strategic Impact |
| :--- | :--- | :--- |
| **Undervalued "Bargain" Opportunities** | **47 homes (16.1%)** | Properties priced ≥10% below model fair value (*Buy Candidates*) |
| **Gross Arbitrage Profit Potential** | **$1,572,846.41** | Total dollar spread available from acquiring undervalued properties |
| **Average Profit Margin per Arbitrage Deal** | **24.52%** | Average gross upside percentage on identified bargain properties |
| **Overvalued Properties Flagged** | **42 homes (14.4%)** | Properties priced ≥10% above model fair value (*Capital Protection / Sell Flags*) |
| **Downside Risk Exposure Mitigated** | **$1,186,870.37** | Total overpayment capital prevented by flagging overvalued assets |

---

## 4. Price Tier Breakdown & Accuracy

| Market Tier | Count | Share (%) | Average Price | Model MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Entry-Level (<$130k)** | 75 | 25.7% | $105,693.55 | 14.10% |
| **Mid-Market ($130k-$250k)** | 166 | 56.8% | $176,672.01 | 7.64% |
| **Premium/Luxury (>$250k)** | 51 | 17.5% | $307,947.27 | 7.42% |

---

## 5. Temporal Trends & Year-over-Year (YoY) Growth

| Year | Transactions (Units) | YoY Volume Growth | Total Market Volume ($) | Average Sale Price ($) | YoY Price Growth |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2006** | 56 | Baseline | $10,481,753.00 | $187,174.16 | Baseline |
| **2007** | 70 | +25.0% | $12,485,348.00 | $178,362.11 | -4.7% |
| **2008** | 57 | -18.6% | $9,660,538.00 | $169,483.12 | -5.0% |
| **2009** | 70 | +22.8% | $14,069,447.00 | $200,992.10 | +18.6% |
| **2010** | 39 | -44.3% | $6,262,794.00 | $160,584.46 | -20.1% |

---

## 6. Executive KPI Summary Visualizations

The visual dashboard has been generated and saved to:
`business_kpi_dashboard.png`
