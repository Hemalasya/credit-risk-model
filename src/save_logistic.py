import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv('data/X_train_scaled.csv', index_col=0)
y_train = pd.read_csv('data/y_train.csv', index_col=0).squeeze()

model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'data/logistic_model.joblib')
joblib.dump(0.70, 'data/logistic_best_threshold.joblib')
print("Saved logistic regression model and threshold to data/")