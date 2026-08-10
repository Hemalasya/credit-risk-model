import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

X_train = pd.read_csv('data/X_train_encoded.csv', index_col=0)
X_val = pd.read_csv('data/X_val_encoded.csv', index_col=0)

scaler = StandardScaler()

# fit_transform on train: learns mean/std from train AND applies it in one step
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# transform ONLY on val: applies train's already-learned mean/std, does NOT relearn from val
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)

print("Train scaled — mean should be ~0, std should be ~1:")
print(X_train_scaled.describe().loc[['mean', 'std']].T.head())

print("\nVal scaled — mean will NOT be exactly 0 (that's expected and correct,")
print("since val is scaled using train's mean/std, not its own):")
print(X_val_scaled.describe().loc[['mean', 'std']].T.head())

X_train_scaled.to_csv('data/X_train_scaled.csv')
X_val_scaled.to_csv('data/X_val_scaled.csv')

# Save the fitted scaler itself — you'll need this exact same scaler later
# to transform any new data (e.g., the test set, or live predictions in your API)
joblib.dump(scaler, 'data/scaler.joblib')
print("\nSaved scaled splits and fitted scaler to data/")