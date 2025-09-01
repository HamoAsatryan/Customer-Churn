import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import SMOTE

df = pd.read_csv('/home/subaru/customer-churn/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop("customerID", axis=1, inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    if col != "Churn":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

param_grids = {
    "Random Forest": {
        'n_estimators': [200, 500],
        'max_depth': [15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "XGBoost": {
        'n_estimators': [100, 200],
        'max_depth': [7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.9, 1]
    }
}

results = {}

for name, model in models.items():
    print(f'\n=== {name} ===')
    if name in param_grids:
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_res, y_train_res)
        best_model = grid.best_estimator_
        print("Best params:", grid.best_params_)
    else:
        model.fit(X_train_res, y_train_res)
        best_model = model

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": best_model,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba
    }

import joblib

best_model = results["Logistic Regression"]["model"]
joblib.dump(best_model, "../models/churn_model.joblib")