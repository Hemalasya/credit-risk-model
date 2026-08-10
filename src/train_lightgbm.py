import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

X_train = pd.read_csv('data/X_train_encoded.csv', index_col=0)
X_val = pd.read_csv('data/X_val_encoded.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0).squeeze()
y_val = pd.read_csv('data/y_val.csv', index_col=0).squeeze()

# Equivalent of class_weight='balanced' for LightGBM
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

model = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=200
)
model.fit(X_train, y_train)

y_val_proba = model.predict_proba(X_val)[:, 1]
y_val_pred = model.predict(X_val)

print("\nROC-AUC:", roc_auc_score(y_val, y_val_proba))
print("\nClassification report:")
print(classification_report(y_val, y_val_pred))

print("Confusion matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("(rows = actual, columns = predicted; order is [no-default, default])")

# Feature importance — LightGBM's equivalent of logistic regression's coefficients,
# though interpreted differently: this measures how often/how usefully a feature
# was used to split the trees, not a signed direction of effect
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nTop 10 features by importance:")
print(importances.head(10))