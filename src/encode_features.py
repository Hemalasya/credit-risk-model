import pandas as pd

X_train = pd.read_csv('data/X_train_features.csv', index_col=0)
X_val = pd.read_csv('data/X_val_features.csv', index_col=0)

# One-hot encode utilization_band on train — this defines the official column set
X_train_encoded = pd.get_dummies(X_train, columns=['utilization_band'], prefix='util')

# Encode validation the same way...
X_val_encoded = pd.get_dummies(X_val, columns=['utilization_band'], prefix='util')

# ...then force validation to have exactly train's columns.
# reindex adds any missing column (filled with 0) and drops any extra one,
# so both sets end up with identical columns in identical order.
X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

print("Train columns:", list(X_train_encoded.columns))
print("\nTrain shape:", X_train_encoded.shape)
print("Val shape:", X_val_encoded.shape)
print("\nColumns match exactly:", list(X_train_encoded.columns) == list(X_val_encoded.columns))

X_train_encoded.to_csv('data/X_train_encoded.csv')
X_val_encoded.to_csv('data/X_val_encoded.csv')
print("\nSaved encoded splits to data/")