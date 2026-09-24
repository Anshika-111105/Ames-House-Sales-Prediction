import os
import sys
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

project_root = Path(__file__).resolve().parent.parent

def set_cell_shading(cell, color_hex: str):
    """Applies background color shading to a table cell."""
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell internal padding."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_project_report():
    doc = docx.Document()
    
    # Page setup - Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Define Palette
    PRIMARY_COLOR = RGBColor(31, 78, 121)     # Deep Navy
    SECONDARY_COLOR = RGBColor(46, 117, 182)  # Steel Blue
    ACCENT_COLOR = RGBColor(16, 124, 65)      # Forest Green
    DARK_TEXT = RGBColor(38, 38, 38)          # Off-black
    
    # Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = DARK_TEXT
    
    # ============================================================
    # COVER / HEADER TITLE
    # ============================================================
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(10)
    title_p.paragraph_format.space_after = Pt(4)
    run_title = title_p.add_run("Ames Real Estate Valuation & Decision Analytics Pipeline")
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = PRIMARY_COLOR
    
    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_p.paragraph_format.space_after = Pt(18)
    run_sub = subtitle_p.add_run("End-to-End Machine Learning, Two-Tier KPI Evaluation & Strategic Decision Support")
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = SECONDARY_COLOR
    
    # Metadata Block
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    row0 = meta_table.rows[0].cells
    row0[0].paragraphs[0].add_run("Author: ").bold = True
    row0[0].paragraphs[0].add_run("Anshika Saklani")
    row0[1].paragraphs[0].add_run("Repository: ").bold = True
    row0[1].paragraphs[0].add_run("Ames-House-Sales-Prediction")
    
    row1 = meta_table.rows[1].cells
    row1[0].paragraphs[0].add_run("Domain: ").bold = True
    row1[0].paragraphs[0].add_run("Real Estate Valuation & PropTech")
    row1[1].paragraphs[0].add_run("Model Framework: ").bold = True
    row1[1].paragraphs[0].add_run("Scikit-Learn Regularized Regressors (Lasso α=50.0)")
    
    for row in meta_table.rows:
        for cell in row.cells:
            set_cell_shading(cell, "F2F4F8")
            set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
            
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Helper to add section headings
    def add_heading_1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = PRIMARY_COLOR
        return h
        
    def add_heading_2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.size = Pt(12.5)
        r.font.bold = True
        r.font.color.rgb = SECONDARY_COLOR
        return h

    # ============================================================
    # 1. EXECUTIVE SUMMARY
    # ============================================================
    add_heading_1("1. Executive Summary")
    doc.add_paragraph(
        "Accurate property valuation is fundamental to residential lending, portfolio investment, and property acquisition. "
        "Traditional human appraisals are slow, expensive ($300–$500 per valuation), and vulnerable to subjective appraisal bias. "
        "This project presents an enterprise-grade, end-to-end Machine Learning and Decision Support Pipeline trained on the "
        "comprehensive Ames Housing Dataset (1,460 historic sales with 79 structural, spatial, and qualitative features)."
    )
    doc.add_paragraph(
        "By packaging custom feature engineering, median imputation, categorical one-hot encoding, and standard scaling into a "
        "leakage-free Scikit-learn Pipeline, our tuned Lasso Regression model achieves an R² of 91.84% with a Mean Absolute Error (MAE) "
        "of $15,182.56. Crucially, the project goes beyond statistical accuracy by introducing a Two-Tier KPI Framework and an actionable "
        "Decision & Recommendation Engine that classifies properties into acquisition actions (STRONG BUY, FAIR VALUE, PASS/NEGOTIATE) "
        "and estimates ROI on value-add renovation strategies."
    )
    
    # ============================================================
    # 2. PROBLEM STATEMENT & OBJECTIVES
    # ============================================================
    add_heading_1("2. Problem Statement & Project Objectives")
    add_heading_2("2.1 Real-World Problem Definition")
    
    p_prob = doc.add_paragraph()
    p_prob.add_run("• What Real-World Problem is Solved? ").bold = True
    p_prob.add_run("Real estate pricing opacity creates financial friction, appraisal delays, and mispriced property purchases.\n")
    p_prob.add_run("• Who Faces This Problem? ").bold = True
    p_prob.add_run("Institutional real estate investors (REITs), mortgage underwriters, PropTech platforms, home buyers, and appraisers.\n")
    p_prob.add_run("• Why Does It Matter? ").bold = True
    p_prob.add_run("Residential real estate is the largest global asset class. Even a 5% valuation error on an average home ($180k) causes a $9,000 capital misallocation or increased loan default risk.\n")
    p_prob.add_run("• Decisions Supported: ").bold = True
    p_prob.add_run("Automated fair market appraisal, deal acquisition filtering, counter-offer price caps, and property renovation ROI prioritization.")

    add_heading_2("2.2 Concrete Project Objectives")
    for obj in [
        "1. Ingest, clean, and analyze 79 residential features from the Ames Housing Dataset with zero data leakage.",
        "2. Engineer domain-specific composite metrics including HouseAge, TotalLivingArea, TotalBathrooms, and TotalPorchArea.",
        "3. Systematically evaluate and cross-validate baseline OLS, Ridge, and Lasso regressors with/without polynomial feature interactions.",
        "4. Establish a Two-Tier KPI Framework that balances ML statistical accuracy (R², MAE, RMSE) with enterprise business metrics (Asset Scale, Appraisal OPEX Savings, Precision Tolerance Bands, Arbitrage Profit Potential).",
        "5. Deploy an automated Decision & Recommendation Engine for deal scoring and renovation ROI simulation.",
        "6. Enforce software engineering reliability through a 100% passing automated test suite (Pytest)."
    ]:
        doc.add_paragraph(obj, style='List Bullet')

    # ============================================================
    # 3. DATASET & EXPLORATORY DATA ANALYSIS
    # ============================================================
    add_heading_1("3. Dataset Description & Exploratory Data Analysis")
    doc.add_paragraph(
        "The project utilizes the Ames Housing Dataset (compiled by Dean De Cock), covering 1,460 residential property transactions in Ames, Iowa from 2006 to 2010. "
        "The target variable is SalePrice (continuous USD). The dataset contains 37 numerical and 43 categorical predictors describing spatial dimensions, "
        "quality ratings, construction dates, and neighborhood location."
    )
    doc.add_paragraph(
        "Data Hygiene & Outlier Handling: Following De Cock's published recommendations, extreme outliers with above-ground living area exceeding 4,000 sq ft "
        "and sale prices under $300,000 were filtered out during preprocessing. Missing numerical values were imputed via median statistics computed strictly "
        "on the training folds, while categorical variables were mode-imputed and one-hot encoded with handle_unknown='ignore'."
    )
    
    # Correlation Heatmap Image
    corr_img = project_root / "outputs" / "correlation_heatmap.png"
    if corr_img.exists():
        add_heading_2("3.1 Feature Correlation Analysis")
        doc.add_picture(str(corr_img), width=Inches(5.8))
        caption = doc.add_paragraph("Figure 1: Target Correlation Heatmap for Top Numerical Drivers of Sale Price.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(10)
        
    doc.add_paragraph(
        "Key EDA Findings: Overall Material & Finish Quality (OverallQual, r=0.79), Above-Grade Living Area (GrLivArea, r=0.71), Total Basement Square Footage "
        "(TotalBsmtSF, r=0.61), Garage Capacity (GarageCars, r=0.64), and Construction Year (YearBuilt, r=0.52) exhibit the strongest direct linear correlations with SalePrice."
    )

    # ============================================================
    # 4. ARCHITECTURE & PIPELINE
    # ============================================================
    add_heading_1("4. Machine Learning Pipeline Architecture")
    doc.add_paragraph(
        "The end-to-end architecture is encapsulated into a modular Scikit-learn Pipeline, ensuring zero data leakage between training, validation, and inference:"
    )
    
    arch_img = project_root / "project_architecture.png"
    if arch_img.exists():
        doc.add_picture(str(arch_img), width=Inches(5.8))
        caption = doc.add_paragraph("Figure 2: End-to-End Pipeline Architecture Flow.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(10)

    # ============================================================
    # 5. MODEL EVALUATION & COMPARATIVE RESULTS
    # ============================================================
    add_heading_1("5. Model Evaluation & Benchmark Results")
    doc.add_paragraph(
        "Six regression configurations were systematically benchmarked on an 80/20 train-validation split (1,166 train rows, 292 validation rows) "
        "and 5-fold cross-validation:"
    )
    
    # Table of Model Comparison
    models_data = [
        ("Linear Regression (Base)", "No", "No", "0.7049", "$18,350.91", "$40,372.14", "$26,104.82"),
        ("Ridge Regression (Tuned)", "No", "No", "0.9182", "$15,339.82", "$21,253.62", "$23,110.09"),
        ("Lasso Regression (Tuned)", "No", "No", "0.9184", "$15,182.56", "$21,228.34", "$23,679.02"),
        ("Linear Reg + Poly + SelectKBest", "Yes", "k=50", "0.8675", "$19,271.66", "$27,048.72", "$27,118.76"),
        ("Ridge + Poly + SelectKBest", "Yes", "k=50", "0.8692", "$19,246.23", "$26,880.60", "$26,807.45"),
        ("Lasso + Poly + SelectKBest", "Yes", "k=50", "0.8644", "$19,456.47", "$27,368.44", "$26,787.28")
    ]
    
    table = doc.add_table(rows=len(models_data) + 1, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    headers = ["Model Architecture", "Poly", "SelectKBest", "Val R²", "Val MAE", "Val RMSE", "CV RMSE"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.paragraphs[0].add_run(h).bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, "1F4E79")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        
    for row_idx, row_vals in enumerate(models_data, start=1):
        bg = "F9FAFB" if row_idx % 2 == 0 else "FFFFFF"
        if "Lasso Regression (Tuned)" in row_vals[0]:
            bg = "E2F0D9"  # Highlight Best
        for col_idx, val in enumerate(row_vals):
            cell = table.cell(row_idx, col_idx)
            r = cell.paragraphs[0].add_run(val)
            if "Lasso Regression (Tuned)" in row_vals[0]:
                r.bold = True
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # Model Comparison Chart
    comp_img = project_root / "outputs" / "model_comparison.png"
    if comp_img.exists():
        doc.add_picture(str(comp_img), width=Inches(5.6))
        caption = doc.add_paragraph("Figure 3: Validation RMSE Comparison Across All 6 Regression Models.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(10)
        
    # Pred vs Actual and Residuals
    pred_img = project_root / "outputs" / "pred_vs_actual.png"
    res_img = project_root / "outputs" / "residuals.png"
    if pred_img.exists() and res_img.exists():
        add_heading_2("5.1 Model Diagnostics & Error Analysis")
        doc.add_picture(str(pred_img), width=Inches(5.5))
        caption = doc.add_paragraph("Figure 4: Predicted vs. Actual Sale Price Scatter Plot (Lasso Regressor).")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(6)
        
        doc.add_picture(str(res_img), width=Inches(5.5))
        caption = doc.add_paragraph("Figure 5: Residual Error Distribution on Train and Validation Splits.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(10)

    # ============================================================
    # 6. TWO-TIER KPI FRAMEWORK
    # ============================================================
    add_heading_1("6. Two-Tier KPI Framework: Statistical vs. Commercial Metrics")
    doc.add_paragraph(
        "A critical innovation of this project is the explicit separation between Model KPIs (verifying mathematical fit) "
        "and Business KPIs (quantifying operational utility, asset scale, and investment ROI):"
    )
    
    kpi_dashboard_img = project_root / "outputs" / "business_kpi_dashboard.png"
    if kpi_dashboard_img.exists():
        doc.add_picture(str(kpi_dashboard_img), width=Inches(5.8))
        caption = doc.add_paragraph("Figure 6: Executive Commercial & Operational KPI Dashboard.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(10)
        
    kpis_table_data = [
        ("Valuation Scale (Orders)", "292 properties (Val) / 1,459 (Inference)", "Total deal volume processed by the automated valuation model"),
        ("Portfolio Asset Volume (Revenue)", "$52.96 Million (Val) / $261.28 Million (Test)", "Aggregate dollar volume of residential real estate assessed"),
        ("Average Property Valuation (AOV)", "$181,722.94", "Mean transaction valuation baseline"),
        ("Appraisal Cost Savings", "$116,800 (Val) / $583,600 (Inference)", "Operational OPEX saved at $400/manual appraisal automated"),
        ("Underwriting Hours Saved", "1,168 hours (Val) / 5,836 hours (Test)", "Turnaround time eliminated (4 hours per manual appraisal)"),
        ("Within ±10% Accuracy SLA", "69.52% of properties", "High-precision commercial automated valuation standard"),
        ("Within ±20% Accuracy SLA", "94.18% of properties", "Safe mortgage underwriting threshold"),
        ("Arbitrage Opportunities (Buy Flags)", "47 homes (16.1% of portfolio)", "Properties priced ≥10% below model fair value"),
        ("Gross Arbitrage Profit Spread", "$1,572,846.41", "Total dollar spread available from acquiring undervalued properties")
    ]
    
    kpi_table = doc.add_table(rows=len(kpis_table_data) + 1, cols=3)
    kpi_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    kpi_table.autofit = False
    
    kpi_headers = ["Commercial KPI", "Achieved Value", "Business Impact"]
    for col_idx, h in enumerate(kpi_headers):
        cell = kpi_table.cell(0, col_idx)
        cell.paragraphs[0].add_run(h).bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, "2E75B6")
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        
    for row_idx, row_vals in enumerate(kpis_table_data, start=1):
        bg = "F9FAFB" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, val in enumerate(row_vals):
            cell = kpi_table.cell(row_idx, col_idx)
            r = cell.paragraphs[0].add_run(val)
            if col_idx == 0:
                r.bold = True
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # ============================================================
    # 7. EXPLAINABILITY & DECISION SUPPORT
    # ============================================================
    add_heading_1("7. Explainability & Decision Support Engine")
    
    feat_img = project_root / "outputs" / "feature_importance.png"
    if feat_img.exists():
        add_heading_2("7.1 Model Explainability & Key Value Drivers")
        doc.add_picture(str(feat_img), width=Inches(5.6))
        caption = doc.add_paragraph("Figure 7: Absolute Feature Importance / Coefficient Weights of the Lasso Model.")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(10)
        
    doc.add_paragraph(
        "Top Valuation Drivers: Total Living Area (above grade + basement), Overall Quality, Neighborhood (e.g. Northridge Heights, Stone Brook), "
        "and Year Built carry the highest positive coefficient weights. In contrast, older properties and deferred maintenance penalize valuation."
    )
    
    add_heading_2("7.2 Deal Classification & Renovation ROI Simulator")
    doc.add_paragraph(
        "Our Decision Engine (src/recommendation_engine.py) applies programmatic rules to translate price predictions into tactical actions:"
    )
    doc.add_paragraph("• STRONG BUY (16.1% of homes): Price is ≥10% below model fair value. Recommended for immediate acquisition to capture equity spread.", style='List Bullet')
    doc.add_paragraph("• FAIR VALUE (69.5% of homes): Price is within ±10% of fair value. Approved for standard mortgage underwriting.", style='List Bullet')
    doc.add_paragraph("• PASS / NEGOTIATE (14.4% of homes): Price is ≥10% above fair value. Flags capital risk and generates counter-offer price ceiling.", style='List Bullet')
    
    doc.add_paragraph(
        "Renovation ROI Advisor: On a representative median home ($163,000 baseline), adding 300 sq ft of finished basement space yields an estimated value lift of $14,200 "
        "(net ROI: +42%), while upgrading Overall Quality by +1 point yields a value lift of $16,800 (net ROI: +40%)."
    )

    # ============================================================
    # 8. TESTING & QUALITY ASSURANCE
    # ============================================================
    add_heading_1("8. Automated Testing & Reliability")
    doc.add_paragraph(
        "To satisfy enterprise production standards, a comprehensive automated test suite was constructed using Pytest (tests/test_pipeline.py):"
    )
    doc.add_paragraph("1. Data Validation Tests: Verify presence of train/test CSVs, required schema columns, non-null targets, and positive pricing.", style='List Bullet')
    doc.add_paragraph("2. Feature Engineering Tests: Validate dynamic creation of HouseAge, TotalBathrooms, and TotalLivingArea with boundary capping.", style='List Bullet')
    doc.add_paragraph("3. Model Pipeline Tests: Verify joblib deserialization, input batch processing, and non-negative prediction output shape.", style='List Bullet')
    doc.add_paragraph("4. Business KPI Tests: Ensure mathematical accuracy of appraisal savings, precision bands, and portfolio totals.", style='List Bullet')
    doc.add_paragraph("5. Decision Rule Tests: Test decision boundaries for STRONG BUY, FAIR VALUE, and PASS/NEGOTIATE classifications.", style='List Bullet')
    doc.add_paragraph("Test Results: 6/6 tests passing (100% pass rate in 3.81 seconds).", style='List Bullet').runs[0].bold = True

    # ============================================================
    # 9. CONCLUSION & FUTURE SCOPE
    # ============================================================
    add_heading_1("9. Conclusion & Future Improvements")
    doc.add_paragraph(
        "This project successfully delivers an end-to-end Machine Learning, Two-Tier KPI, and Decision Analytics solution for real estate valuation. "
        "By grounding complex regularized linear models in concrete commercial metrics and actionable decision frameworks, it bridges the gap between "
        "data science algorithms and real-world business execution."
    )
    doc.add_paragraph(
        "Future Enhancements: (1) Integrate gradient boosting ensembles (XGBoost / LightGBM) for non-linear residual benchmarking; "
        "(2) Introduce automated target transforms (log1p) to further refine high-value luxury property predictions; "
        "(3) Incorporate spatial GIS coordinates for precise neighborhood distance decay modeling."
    )
    
    output_path = project_root / "Ames_House_Sales_Prediction_Project_Report.docx"
    doc.save(str(output_path))
    print(f"Successfully created project report at {output_path}")

if __name__ == "__main__":
    create_project_report()
