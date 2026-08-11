from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load everything once at startup, not per-request — loading from disk on every
# request would be slow and pointless since none of this changes between requests
model = joblib.load('data/logistic_model.joblib')
scaler = joblib.load('data/scaler.joblib')
fit_values = joblib.load('data/fit_values.joblib')
threshold = joblib.load('data/logistic_best_threshold.joblib')

late_cols = ['NumberOfTime30-59DaysPastDueNotWorse',
             'NumberOfTime60-89DaysPastDueNotWorse',
             'NumberOfTimes90DaysLate']

# The exact ordered list of columns the scaler/model expect, post-encoding.
# Hardcoded here because a single incoming applicant may not naturally produce
# all 4 utilization_band dummy columns (e.g., if their band is 'low', the
# 'high'/'moderate'/'very_low' columns wouldn't exist unless we force them).
EXPECTED_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents',
    'MonthlyIncome_missing', 'late_payment_data_unknown', 'disposable_income',
    'delinquency_score', 'credit_lines_per_dependent',
    'util_high', 'util_low', 'util_moderate', 'util_very_low'
]

def preprocess(applicant: dict) -> pd.DataFrame:
    df = pd.DataFrame([applicant])

    # --- Cleaning (same logic as clean_data.py, using saved fit_values) ---
    df['MonthlyIncome_missing'] = df['MonthlyIncome'].isnull().astype(int)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(fit_values['income_median'])
    df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=fit_values['income_cap'])
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(fit_values['dependents_fill'])

    df['late_payment_data_unknown'] = df[late_cols[0]].isin([96, 98]).astype(int)
    for col in late_cols:
        df.loc[df[col].isin([96, 98]), col] = fit_values['late_median']

    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=fit_values['util_cap'])
    df['DebtRatio'] = df['DebtRatio'].clip(upper=fit_values['debt_cap'])

    # --- Feature engineering (same logic as feature_engineering.py) ---
    df['disposable_income'] = df['MonthlyIncome'] * (1 - df['DebtRatio'].clip(upper=1))
    df['delinquency_score'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
        df['NumberOfTime60-89DaysPastDueNotWorse'] * 2 +
        df['NumberOfTimes90DaysLate'] * 3
    )
    df['utilization_band'] = pd.cut(
        df['RevolvingUtilizationOfUnsecuredLines'],
        bins=[-0.01, 0.1, 0.3, 0.5, 1.5],
        labels=['very_low', 'low', 'moderate', 'high']
    )
    df['credit_lines_per_dependent'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['NumberOfDependents'] + 1)

    # --- Encoding (same logic as encode_features.py) ---
    df = pd.get_dummies(df, columns=['utilization_band'], prefix='util')
    # Force all expected columns to exist, filling any missing dummy with 0 —
    # same reindex trick used in encode_features.py for train/val consistency
    df = df.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

    # --- Scaling (using the already-fitted scaler) ---
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df_scaled

@app.route('/predict', methods=['POST'])
def predict():
    applicant = request.get_json()
    X = preprocess(applicant)

    probability = model.predict_proba(X)[0, 1]
    prediction = int(probability >= threshold)

    return jsonify({
        'default_probability': round(float(probability), 4),
        'prediction': prediction,
        'prediction_label': 'high_risk' if prediction == 1 else 'low_risk',
        'threshold_used': threshold
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)