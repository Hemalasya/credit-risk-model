import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

X_train = pd.read_csv('data/X_train_scaled.csv', index_col=0)
X_val = pd.read_csv('data/X_val_scaled.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0).squeeze()
y_val = pd.read_csv('data/y_val.csv', index_col=0).squeeze()

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predicted probabilities (needed for ROC-AUC) and hard 0/1 predictions (needed for precision/recall)
y_val_proba = model.predict_proba(X_val)[:, 1]
y_val_pred = model.predict(X_val)

print("ROC-AUC:", roc_auc_score(y_val, y_val_proba))
print("\nClassification report:")
print(classification_report(y_val, y_val_pred))

print("Confusion matrix:")
print(confusion_matrix(y_val, y_val_pred))
print("(rows = actual, columns = predicted; order is [no-default, default])")

# Feature coefficients — this is your interpretability payoff, ranked by impact
coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
print("\nTop 10 features by coefficient magnitude:")
print(coefs.head(10))