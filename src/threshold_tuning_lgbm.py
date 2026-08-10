import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import confusion_matrix

X_train = pd.read_csv('data/X_train_encoded.csv', index_col=0)
X_val = pd.read_csv('data/X_val_encoded.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0).squeeze()
y_val = pd.read_csv('data/y_val.csv', index_col=0).squeeze()

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, n_estimators=200, verbose=-1)
model.fit(X_train, y_train)
y_val_proba = model.predict_proba(X_val)[:, 1]

FN_COST = 5
FP_COST = 1

results = []
for threshold in np.arange(0.05, 0.95, 0.05):
    y_pred = (y_val_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    total_cost = (FN_COST * fn) + (FP_COST * fp)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    results.append({
        'threshold': round(threshold, 2), 'false_negatives': fn, 'false_positives': fp,
        'total_cost': total_cost, 'recall': round(recall, 3), 'precision': round(precision, 3)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best = results_df.loc[results_df['total_cost'].idxmin()]
print(f"\nBest threshold by cost: {best['threshold']} (total cost: {best['total_cost']})")

# Save the tuned model and its chosen threshold together — this is what Project 2's
# Flask API will load later
import joblib
joblib.dump(model, 'data/lightgbm_model.joblib')
joblib.dump(best['threshold'], 'data/lightgbm_best_threshold.joblib')
print("\nSaved LightGBM model and best threshold to data/")