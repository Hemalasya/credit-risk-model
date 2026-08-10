import pandas as pd
from sklearn.model_selection import train_test_split

# --- Load raw data ---
df = pd.read_csv('data/raw/cs-training.csv', index_col=0)
df = df[df['age'] > 0].copy()  # drop the one invalid row — safe before split, doesn't leak any statistic

X = df.drop(columns=['SeriousDlqin2yrs'])
y = df['SeriousDlqin2yrs']

# --- Split FIRST, before computing any cleaning statistics ---
# Stratify on y because the target is imbalanced (6.7% positive) — without stratify,
# a random split could give train/val noticeably different default rates
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train default rate:", y_train.mean())
print("Val default rate:", y_val.mean())

# --- Fit cleaning parameters on X_train ONLY ---
late_cols = ['NumberOfTime30-59DaysPastDueNotWorse',
             'NumberOfTime60-89DaysPastDueNotWorse',
             'NumberOfTimes90DaysLate']

fit_values = {
    'income_median': X_train['MonthlyIncome'].median(),
    'dependents_fill': 0,
    'late_median': X_train[late_cols[0]].where(~X_train[late_cols[0]].isin([96, 98])).median(),
    'util_cap': X_train['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99),
    'debt_cap': X_train['DebtRatio'].quantile(0.99),
    'income_cap': X_train['MonthlyIncome'].quantile(0.99),
}

def clean(df_part, fit_values):
    df_part = df_part.copy()

    # Missing income: flag + median impute, THEN cap extreme values (order matters —
    # fill NaNs first, so clip() has real numbers to act on everywhere)
    df_part['MonthlyIncome_missing'] = df_part['MonthlyIncome'].isnull().astype(int)
    df_part['MonthlyIncome'] = df_part['MonthlyIncome'].fillna(fit_values['income_median'])
    df_part['MonthlyIncome'] = df_part['MonthlyIncome'].clip(upper=fit_values['income_cap'])

    # Missing dependents: simple impute
    df_part['NumberOfDependents'] = df_part['NumberOfDependents'].fillna(fit_values['dependents_fill'])

    # Sentinel recoding (96/98) in late-payment columns
    df_part['late_payment_data_unknown'] = df_part[late_cols[0]].isin([96, 98]).astype(int)
    for col in late_cols:
        df_part.loc[df_part[col].isin([96, 98]), col] = fit_values['late_median']

    # Cap extreme outliers (winsorize) at the 99th percentile learned from train
    df_part['RevolvingUtilizationOfUnsecuredLines'] = df_part['RevolvingUtilizationOfUnsecuredLines'].clip(upper=fit_values['util_cap'])
    df_part['DebtRatio'] = df_part['DebtRatio'].clip(upper=fit_values['debt_cap'])

    return df_part

X_train_clean = clean(X_train, fit_values)
X_val_clean = clean(X_val, fit_values)

print("\nMissing values remaining in train:", X_train_clean.isnull().sum().sum())
print("Missing values remaining in val:", X_val_clean.isnull().sum().sum())
print("\nX_train_clean shape:", X_train_clean.shape)
print("X_val_clean shape:", X_val_clean.shape)

# --- Save cleaned data so later scripts (feature engineering, modeling) can reuse it ---
X_train_clean.to_csv('data/X_train_clean.csv')
X_val_clean.to_csv('data/X_val_clean.csv')
y_train.to_csv('data/y_train.csv')
y_val.to_csv('data/y_val.csv')
print("\nSaved cleaned splits to data/")
import joblib
joblib.dump(fit_values, 'data/fit_values.joblib')
print("Saved fit_values to data/fit_values.joblib")