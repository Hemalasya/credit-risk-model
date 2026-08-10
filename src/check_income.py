import pandas as pd

X_train = pd.read_csv('data/X_train_clean.csv', index_col=0)
print(X_train['MonthlyIncome'].describe())
print("\nRows with income > 50000:", (X_train['MonthlyIncome'] > 50000).sum())
print("99th percentile:", X_train['MonthlyIncome'].quantile(0.99))