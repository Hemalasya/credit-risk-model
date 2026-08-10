import pandas as pd

# --- Load the cleaned splits from Step 2 ---
X_train = pd.read_csv('data/X_train_clean.csv', index_col=0)
X_val = pd.read_csv('data/X_val_clean.csv', index_col=0)

def add_features(df):
    df = df.copy()

    # 1. Disposable income: actual dollars left after debt obligations
    #    DebtRatio clipped at 1 so we don't get negative "disposable income"
    #    for the extreme (winsorized) debt ratio cases
    df['disposable_income'] = df['MonthlyIncome'] * (1 - df['DebtRatio'].clip(upper=1))

    # 2. Severity-weighted delinquency score: a 90-day-late incident counts
    #    more than a 30-day one, reflecting how lenders actually assess risk
    df['delinquency_score'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
        df['NumberOfTime60-89DaysPastDueNotWorse'] * 2 +
        df['NumberOfTimes90DaysLate'] * 3
    )

    # 3. Utilization risk band: bucket into known real-world risk thresholds
    #    rather than treating utilization as purely linear
    df['utilization_band'] = pd.cut(
        df['RevolvingUtilizationOfUnsecuredLines'],
        bins=[-0.01, 0.1, 0.3, 0.5, 1.5],
        labels=['very_low', 'low', 'moderate', 'high']
    )

    # 4. Credit lines per dependent: financial burden relative to household size
    #    (+1 in denominator avoids divide-by-zero for people with 0 dependents)
    df['credit_lines_per_dependent'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['NumberOfDependents'] + 1)

    return df

X_train_feat = add_features(X_train)
X_val_feat = add_features(X_val)

print("Train shape after feature engineering:", X_train_feat.shape)
print("Val shape after feature engineering:", X_val_feat.shape)
print("\nNew columns added:")
print(set(X_train_feat.columns) - set(X_train.columns))

print("\nSample of new features:")
print(X_train_feat[['disposable_income', 'delinquency_score', 'utilization_band', 'credit_lines_per_dependent']].head())

# --- Save for Step 4 (modeling) ---
X_train_feat.to_csv('data/X_train_features.csv')
X_val_feat.to_csv('data/X_val_features.csv')
print("\nSaved feature-engineered splits to data/")