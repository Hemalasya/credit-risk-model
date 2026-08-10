content = '''# Credit Risk Prediction Model

Predicting loan applicant default risk using the "Give Me Some Credit" dataset (150,000 applicants), with a focus on cost-aware decision-making rather than raw accuracy.

## Problem

Build a model to flag loan applicants likely to become seriously delinquent (90+ days past due) within 2 years, where the cost of approving a defaulter is significantly higher than the cost of rejecting a creditworthy applicant. The goal isn't just prediction accuracy - it's identifying a decision threshold that reflects that cost asymmetry.

## Data quality findings

Rather than treating this as a clean dataset, I investigated several data quality issues before modeling:

- **19.8% of rows missing MonthlyIncome** - imputed with the training-set median, plus a binary MonthlyIncome_missing flag so the model can use missingness itself as a signal.
- **DebtRatio and RevolvingUtilizationOfUnsecuredLines contained extreme outliers** (max values of 329,664 and 50,708 respectively, for fields that should behave like ratios). Winsorized (capped) at the 99th percentile rather than dropped, since the affected rows were too large a share of the data (11.3% for DebtRatio) to discard.
- **The three days-past-due columns contained a 96/98 sentinel code** in 269 rows, all overlapping across all three columns simultaneously - strong evidence of a placeholder/error code rather than real values. These rows had a 54.6% actual default rate vs. 6.7% overall, so instead of dropping them, I recoded the sentinel into a late_payment_data_unknown flag and imputed the underlying count with the training median, preserving these high-signal rows.
- **MonthlyIncome outliers surfaced downstream**: after capping DebtRatio and utilization but not income, an engineered disposable_income feature produced an absurd value (67,430/month) for one applicant. Traced this back to an uncapped income column and fixed it at the source rather than patching the derived feature.

All cleaning statistics (medians, percentile caps) were fit on the training split only and applied to validation, to avoid data leakage.

## Feature engineering

Beyond the raw columns, engineered 4 features reflecting how a credit analyst actually reasons about risk:
- disposable_income - income remaining after debt obligations
- delinquency_score - late payments weighted by severity (30/60/90-day categories weighted 1/2/3)
- utilization_band - credit utilization bucketed into real-world risk thresholds
- credit_lines_per_dependent - financial burden relative to household size

delinquency_score ranked as the 2nd most important feature in the logistic regression model (ahead of any single raw late-payment column), validating the severity-weighting approach.

## Modeling

Trained two models on an 80/20 stratified train/validation split:

| Model | ROC-AUC |
|---|---|
| Logistic Regression (class_weight=balanced) | 0.862 |
| LightGBM (scale_pos_weight tuned) | 0.864 |

## Threshold tuning

Default classification threshold (0.5) is arbitrary. Using an illustrative cost assumption - missing a defaulter costs 5x more than wrongly flagging a creditworthy applicant - I swept thresholds from 0.05 to 0.90 and selected the one minimizing total cost.

| | Threshold 0.5 (default) | Threshold 0.70 (cost-optimized) |
|---|---|---|
| Total cost | 7,913 | 6,422 |
| Recall | 0.75 | 0.54 |
| Precision | 0.22 | 0.37 |

Cost-optimizing the threshold reduced total cost by ~19% versus the default cutoff.

## Model selection: logistic regression over LightGBM

After tuning both models to their optimal thresholds, LightGBM's total cost (6,370) was only 0.8% better than logistic regression's (6,422) - well within noise, and far too small to justify the loss of interpretability. I recommend logistic regression for production: in lending specifically, explainability isn't a nice-to-have - regulators often require lenders to justify individual credit decisions, which a linear model supports directly through its coefficients and a gradient-boosted model does not.

## Limitations

- The 5:1 cost ratio is an illustrative assumption, not derived from real institutional loss data. A production deployment would calibrate this from actual default/collections costs.
- No temporal validation - the dataset has no time dimension, so this does not test for concept drift.

## How to run

pip install -r requirements.txt
python src/clean_data.py
python src/feature_engineering.py
python src/encode_features.py
python src/scale_features.py
python src/train_logistic.py
python src/threshold_tuning.py
python src/train_lightgbm.py
python src/threshold_tuning_lgbm.py
python src/save_logistic.py
'''

with open('README.md', 'w', encoding='utf-8', newline='\n') as f:
    f.write(content)

print("README.md rewritten cleanly, no escape characters")