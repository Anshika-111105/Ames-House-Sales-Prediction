# Business Interpretation Report

## 1. Chosen Model
The model selected for production is **Lasso Regression (Tuned)**. Through systematic evaluation of Linear Regression, Ridge, and Lasso models, with and without polynomial feature expansions, we found that the tuned **Lasso Regression** model (regularization strength $\alpha = 50.0$) yields the best balance between predictive accuracy and model simplicity. It achieves a Validation $R^2$ of **0.9184**, meaning it explains approximately **91.84%** of the variance in house sale prices in Ames, Iowa.

## 2. Average Prediction Error
- **Mean Absolute Error (MAE):** $15,182.56
- **Root Mean Squared Error (RMSE):** $21,228.34

On average, our model's predictions deviate from the actual sales price by **$15,182.56**. This is an exceptionally strong result for a linear model on this dataset, indicating high reliability for automated valuations of typical residential properties.

## 3. Largest Residuals (Outliers)
Below are the top 5 properties in the validation set with the largest prediction errors:

| Original Row Index | Actual Sale Price | Predicted Price | Absolute Error | Percentage Error |
| :---: | :---: | :---: | :---: | :---: |
| 218 | $311,500.00 | $229,722.44 | $81,777.56 | 26.25% |
| 261 | $276,000.00 | $347,604.16 | $71,604.16 | 25.94% |
| 1173 | $200,500.00 | $270,555.37 | $70,055.37 | 34.94% |
| 608 | $359,100.00 | $294,346.82 | $64,753.18 | 18.03% |
| 271 | $241,500.00 | $180,955.32 | $60,544.68 | 25.07% |

### Analysis of Errors:
Most of the large errors occur on high-end luxury homes (e.g., actual prices exceeding $350,000) or properties with unique structural/qualitative characteristics not fully captured by linear patterns. Under-predicting extremely high-priced outliers is a known limitation of regularized linear models because they penalize extreme coefficient weights.

## 4. Model Limitations
1. **Underprediction of High-Value Properties:** As observed in the residual analysis and the largest residuals table, the model tends to underpredict homes worth over $350,000.
2. **Assumption of Linearity:** Despite adding custom feature engineering, some highly non-linear relationships (e.g., complex neighborhood interactions) might be missed by a linear/Lasso model.
3. **Data Limitations:** The model is trained on historic transaction data from Ames, Iowa (2006–2010) and does not account for macroeconomic shifts, inflation, or different geographical markets.

## 5. Key Value Drivers & Business Recommendations
Based on the Lasso model coefficients, the top 5 positive drivers of property value in Ames are:
1. **Total Living Area (GrLivArea + TotalBsmtSF):** Every additional square foot adds substantial value. Encouraging home remodeling to finish basements or add livable space is highly profitable.
2. **Overall Material & Finish Quality (OverallQual):** Qualitative ratings have a major impact. Moving a house's quality rating up by 1 point (on a 1-10 scale) can raise its value significantly.
3. **Year Built & Age (HouseAge):** Newer houses command a strong premium, which slowly depreciates over time.
4. **Total Bathrooms:** The number of bathrooms remains a critical feature for buyers.
5. **Neighborhood location:** Highly desirable neighborhoods (e.g., Northridge Heights, Stone Brook) add a massive location-based premium.

**Actionable Strategy:**
- **Automated Valuation:** Deploy this pipeline to provide instant baseline valuations for residential properties in Ames.
- **Investment Screening:** Identify underpriced homes by looking for properties where the actual price is significantly lower than the model's predicted price (large positive residuals).
- **Renovation Decisions:** Prioritize improvements in overall materials/finishes and increasing living area square footage, as these yield the highest return on investment.
