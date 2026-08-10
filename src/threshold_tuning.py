import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X_train = pd.read_csv('data/X_train_scaled.csv', index_col=0)
X_val = pd.read_csv('data/X_val_scaled.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0).squeeze()
y_val = pd.read_csv('data/y_val.csv', index_col=0).squeeze()

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_val_proba = model.predict_proba(X_val)[:, 1]

FN_COST = 5   # missed defaulter
FP_COST = 1   # wrongly flagged good applicant

results = []
for threshold in np.arange(0.05, 0.95, 0.05):
    y_pred = (y_val_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    total_cost = (FN_COST * fn) + (FP_COST * fp)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    results.append({
        'threshold': round(threshold, 2),
        'false_negatives': fn,
        'false_positives': fp,
        'total_cost': total_cost,
        'recall': round(recall, 3),
        'precision': round(precision, 3)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best = results_df.loc[results_df['total_cost'].idxmin()]
print(f"\nBest threshold by cost: {best['threshold']} (total cost: {best['total_cost']})")
print(f"Compare to default 0.5 threshold cost: {results_df[results_df['threshold']==0.5]['total_cost'].values[0]}")