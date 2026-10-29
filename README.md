# Big Mart Sales Prediction — AMBA Project

Practice problem: [Big Mart Sales III (Analytics Vidhya)](https://www.analyticsvidhya.com/datahack/contest/practice-problem-big-mart-sales-iii/)

## Problem
BigMart collected 2013 sales data for **1,559 products across 10 stores**. The task is to predict `Item_Outlet_Sales`, the sales of a product in a particular store. Because the target is a continuous number, this is a **regression** problem.

- **Metric:** RMSE (root mean squared error). Lower is better.
- **Submission:** a CSV with columns `Item_Identifier, Outlet_Identifier, Item_Outlet_Sales`, 5,681 rows in the same order as the test file.
- **Goal:** about **rank 300** on the leaderboard, which is roughly **1148 RMSE**.

## Files
| File | Description |
|---|---|
| `train_v9rqX0R.csv` | Training data: 8,523 rows, 11 features plus the target |
| `test_AbJTz2l.csv` | Test data: 5,681 rows with no target |
| `sample_submission_8RXa3c6 (1).csv` | Submission format template |
| `LAB EDA  5.ipynb` | Professor's EDA format reference (Boston dataset) |
| `BigMart_Sales_Prediction.ipynb` | **Main project notebook**: EDA, cleaning, model selection, diagnostics and conclusions |
| `BigMart_Sales_Prediction.py` | Standalone Python export of all notebook cells, with both helper modules bundled; suitable for a single code-file upload |
| `train_clean.csv` / `test_clean.csv` | Cleaned and encoded data produced by the notebook |
| `preprocessing.py` | Training-fitted cleaning and feature engineering, reusable inside sklearn pipelines |
| `feature_columns.json` | Numeric predictor names, excluding identifiers and all target-derived columns |
| `test_preprocessing.py` | Checks for fold refitting, unseen-category fallbacks, batch independence and export integrity |
| `modeling.py` | Raw-input modelling pipelines, seeded CV/searches, frozen evaluation, final refit and inference |
| `test_modeling.py` | Split, feature-profile, scaling, target-transform, model, serialization and submission tests |
| `model_outputs/` | Search results, frozen selection, split assignments, diagnostics, fitted models and run metadata |
| `submission.csv` | Validated 5,681-row submission, in original test order |
| `requirements.txt` | Python dependencies |

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook BigMart_Sales_Prediction.ipynb
```

Run all cells from the project folder to regenerate the saved outputs and cleaned CSVs. The raw input CSVs and professor's reference notebook are not changed. Check the reusable preprocessor with:

```bash
.venv/bin/python -m unittest -v test_preprocessing.py test_modeling.py
```

The standalone Python version preserves notebook text as comments, prints tables, and saves all plots. It needs the three original input CSVs and the installed dependencies, without requiring the notebook or local helper files:

```bash
.venv/bin/python BigMart_Sales_Prediction.py --data-dir .
```

Its results are written to `python_export_outputs/`, keeping the existing notebook artifacts separate. Use `--output-dir PATH` to change that folder, or `--show-plots` to also open plot windows. For the competition form, upload `BigMart_Sales_Prediction.py` as the code file and the existing `submission.csv` as the solution file.

## Models
Limited to the course-approved models:
- **Linear Regression** is the baseline.
- **KNN Regressor** and **SVR (RBF kernel)** are comparison models, using scaled features.
- **Random Forest** is a comparison candidate. The final model is selected by development CV, without requiring a particular family to win.

Logistic Regression is excluded because it is a classification model and this is a regression problem.

## Historical benchmarks recorded in the project notes
These are historical references retained from the original notes, not scores achieved or independently verified by this run. Local CV/check RMSE is not directly comparable to a public leaderboard score.

| Source | Best model | Public leaderboard RMSE |
|---|---|---|
| Analytics Vidhya "Top 20" guide | Random Forest (400 trees, depth 6, min leaf 100) | 1152 (mean baseline 1773, Linear Regression 1202) |
| sharmanavika07 (GitHub, rank 300) | Random Forest | 1148.30 (SVR ~1165, KNN ~1154) |
| mmarshall124 (GitHub) | Tuned Random Forest | 1156.9 |
| Top of leaderboard | — | ~1128 |

---

## Plan

### Phase 1 — EDA & data cleaning ✅ done
This follows the professor's EDA format, step by step. The notebook's saved tables, printed results and plots have also been reviewed; the additions below address gaps in the original analysis.

1. **Univariate analysis:** value counts (%) and countplots for the categorical variables; histograms, skewness and boxplots for the numerical ones.
2. **Bivariate analysis:** scatterplots and a correlation heatmap against sales; mean sales grouped by each categorical variable.
3. **Missing value treatment**
   - All learned values use **training features only**; train and test are not combined to fit statistics.
   - `Item_Weight`: same-item training mean → training item-type median → global training median. Four training items have no observed training weight, so fallbacks are necessary.
   - `Outlet_Size`: known training outlet size → training outlet-type mode → global training mode. The filled values are assumptions, not verified store sizes.
   - `Item_Visibility = 0`: flag it and treat it as missing for this candidate treatment; use the item's positive training mean, then outlet-type/global positive medians. Zero being impossible is not established by the data.
   - Keep three indicators recording original weight/size missingness and zero/missing visibility.
4. **Outlier treatment:** diagnose with 3σ, IQR and the 99th percentile. Keep imputed visibility alongside a separate version capped at **training-fitted** IQR limits. Preserve all sales values and rows: an outlier flag alone does not establish an error. Capping benefits remain to be validated.
5. **Variable creation:** clean fat-content labels (plus Non-Edible), `Item_Category`, age in **2013**, visibility relative to a fixed training-only item reference, `MRP_Band` and `Is_Grocery`.
6. **Variable transformation:** compare log/sqrt sales and visibility; encode categories using a training-fitted schema. Target transforms are EDA views only, **excluded from both cleaned exports and the predictor list**.

**Additional EDA and validation**
- Schema, exact duplicates, repeated item/outlet keys, numeric ranges, finite values, metadata consistency and submission-template order.
- Train/test coverage, numeric distributions, missingness rates, unseen categories and category-proportion differences (TVD).
- Missingness by outlet: weights are entirely missing in OUT019/OUT027; sizes are entirely missing in OUT010/OUT017/OUT045.
- Price-band × outlet-type sales and sample-size heatmaps; Pearson/Spearman correlations within outlet types; location/size composition and outlet-level counts, means and medians.
- High-sales tail by outlet type and redundant feature pairs, including duplicate grocery indicators.
- Export assertions for row order, unchanged target, matching feature schemas, no missing/non-finite predictors and no target-derived predictors.

**Key findings**
- `Item_MRP` has the strongest numerical association (correlation 0.57). Its 4 price bands give mean sales of 736 → 1,616 → 2,631 → 3,681. Sales amount also depends on price by definition; this does not establish an effect on units sold.
- Outlet type matters most among the categories: grocery stores average about 340 in sales, supermarkets about 2,000–3,700. OUT027 (Supermarket Type3) is the top store.
- Weight has almost no marginal linear association; item-type mean differences are modest. Neither finding proves that these features have no predictive value.
- Square-root sales is more symmetric (skew 0.24) than raw sales (1.18) or log sales (−0.88). Regression targets need not be normal; choose transformations by cross-validated **original-scale RMSE**.
- Visibility correlates with sales at about −0.13 overall, but only −0.013 to 0.017 within outlet types. Of 144 training visibility values above the fitted IQR cap, 135 are grocery rows; preserve the uncapped alternative and validate the treatment.
- There are only 10 outlets. Type2 and Type3 each occur in one store; store identity, type, size, location and age cannot be interpreted as independent causal drivers here.
- All test items and outlets occur in training, while the item/outlet combinations differ. The test set has no sales labels, so feature-distribution comparisons cannot establish prediction accuracy.

### Checkpoint — reviewed
Phase 1 outputs were reviewed and the modelling plan was approved. The completed modelling phase follows the EDA in the main notebook.

The cleaned exports use all training rows and retain `Item_Identifier` / `Outlet_Identifier` for traceability. `train_clean.csv` contains only the original sales target in addition to the shared features; `feature_columns.json` explicitly lists the numeric predictors. Log/sqrt target views stay in the notebook's separate `target_views` frame.

Current outputs: **8,523 × 54** training CSV, **5,681 × 53** test CSV, and **51 numeric predictors**. Both CSVs include the two identifiers; only training includes the target. No missing predictor values remain.

**For cross-validation, start from the raw CSVs and place a fresh `BigMartPreprocessor()` inside the model pipeline.** Precomputing these full-training cleaned CSVs before splitting would let validation features influence imputation, clipping, ratio references and category discovery. The transformer's `transform()` excludes identifiers and targets automatically; `transform_frame()` retains identifiers for exports. Unknown nominal levels use an all-zero block; missing required columns and unknown ordered levels raise a clear error.

### Phase 2 — Modelling & submission ✅ done

**Validation design**
- Seed 42; an outlet-stratified 80/20 modelling split: **6,818 development rows / 1,705 reserved-check rows**.
- Identical shuffled, seeded five-fold development CV for all candidates. Every pipeline fits imputation, visibility references, category discovery, feature selection and scaling within each training fold.
- The reserved check is used after the model configuration is frozen. Full-data EDA preceded the split, so it is a reserved modelling check rather than a completely untouched final assessment.
- Selection uses mean **original-scale RMSE**, with exact ties broken by prediction/scoring time and model order. Inverse target transformation and the nonnegative prediction floor are included in every score.

**Feature and tuning plan implemented**
- Compact item/store-identity features; descriptive store metadata with ordered or one-hot size/location; MRP × outlet-type interactions for Linear Regression.
- One reference category per selected one-hot block. No raw IDs, target-derived features, duplicate fat/category columns or duplicate grocery indicators.
- KNN/SVR/forest searches sample three feature profiles, four visibility treatments (imputed raw/sqrt/capped and zero-preserving raw) and three targets (raw/sqrt/log1p). Forest also tests the visibility ratio.
- Linear Regression: **12 configurations**; KNN: **16 sampled candidates**; RBF SVR: **20**; Random Forest: **24**. Five folds per candidate, with seed 42 and four concurrent workers; estimator-level parallelism is one.
- Starting baselines remain eligible if a searched model does not improve them. Complete per-candidate/fold results are saved.

**Development results**

| Model | Mean CV RMSE | Fold SD | Best feature profile | Target |
|---|---:|---:|---|---|
| LinearRegression | 1,082.29 | 15.70 | price_interactions | raw |
| RandomForest | 1,099.80 | 16.78 | descriptive_onehot | sqrt |
| SVR | 1,114.38 | 16.68 | descriptive_ordered | log1p |
| KNN | 1,120.96 | 21.76 | descriptive_onehot | raw |
| MeanBaseline | 1,719.31 | 31.58 | none | raw |

The frozen winner is **Linear Regression with price × outlet-type interactions**, using 32 predictors, raw sales and imputed uncapped visibility. It allows price slopes to differ by outlet type while retaining outlet-specific intercept information. The winning pipeline was selected before reserved-check results were calculated; it was not retuned afterward.

**Frozen-model evaluation**
- Pooled development OOF RMSE: **1,082.41**. This differs slightly from the average of the five fold RMSEs.
- Reserved-check RMSE: **1,045.43**; MAE: **727.17**; R²: **0.600**.
- Reserved-check mean-baseline RMSE: **1,653.80**.
- Actual/predicted and residual plots; error tables by outlet, outlet type, MRP band and high-sales segment; ten-repeat permutation importance after selection.

**Final build and reproduction**
The frozen configuration was refitted on all **8,523** labelled rows. `model_outputs/final_model.joblib` contains the entire raw-input estimator, including preprocessing, selected features, scaling, inverse target transform and the zero floor. A save/load round-trip was checked. `submission.csv` has exactly **5,681 rows** and the required columns, with matching test/template identifiers, preserved order and finite nonnegative predictions.

Run the modelling workflow independently:

```bash
.venv/bin/python -m modeling
```

The main notebook calls the same functions. Saved selection/evaluation results are reused only when the training-data/code/package-version fingerprint matches. To deliberately repeat the search, use `run_selection(..., reuse_results=False)`; do not repeatedly tune against the reserved check. Changing code/data invalidates caches. `model_outputs/progress.log` records original search progress.

Raw-input inference from the saved model:

```python
import joblib
import pandas as pd

model = joblib.load("model_outputs/final_model.joblib")
test = pd.read_csv("test_AbJTz2l.csv")
predictions = model.predict(test)  # Original sales scale; nonnegative
```

`model_outputs/` also contains split assignments, full search tables, baseline/final model comparisons, the frozen selection, OOF/check predictions, grouped errors, permutation importance, standalone PNG plots in `model_outputs/figures/`, and metadata with input hashes, versions, selected features and estimator settings.

### Phase 3 — Conclusions ✅ done

- Price and outlet type dominate the frozen model's permutation diagnostics. Store metadata is correlated; these are predictive associations, not causal effects.
- High-sales cases remain difficult: reserved rows above the development sales p90 have RMSE about **2,167**, with average underprediction about **1,870**. The model was not adjusted using this diagnostic.
- Use forecasts to support store/price-segment inventory planning, with additional review of high-sales items. This analysis cannot establish that changing price or visibility causes higher sales.
- Only ten existing stores and a single 2013 cross-section are available. Results do not establish future-year or unseen-store performance. EDA and model search also limit how independent the development results are.
- The local metrics are **not an Analytics Vidhya leaderboard score**. Submission upload and leaderboard evaluation remain manual.

## Status
- [x] Environment setup
- [x] Phase 1: EDA & cleaning
- [x] Checkpoint review
- [x] Phase 2: Modelling & local submission file
- [x] Phase 3: Conclusions
- [ ] Manual leaderboard upload and score recording
