


# IMPORTS

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

import xgboost as xgb
import joblib
from scipy import sparse


# CONFIG
DATA_PATH = "Telco_Customer_Churn.csv"
RANDOM_STATE = 42

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

mlflow.set_experiment("Telecom_Churn_Model_Selection")


# UTILITY FUNCTIONS
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        try:
            y_score = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score)
        except Exception:
            auc = None

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": auc
    }

def tune_model(model, param_grid, X, y):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,                     # BALANCED
        scoring="roc_auc",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_

# PREPROCESSING
def build_preprocessor(cat_cols, num_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

# MAIN
def main():
    print("\n=== TELECOM CHURN — TUNED MODEL SELECTION (BALANCED) ===\n")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found")

    df = pd.read_csv(DATA_PATH)
    print("Data loaded:", df.shape)

    # Clean blanks
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Target cleaning
    df["Churn"] = df["Churn"].astype(str).str.strip().str.capitalize()
    before = len(df)
    df = df.dropna(subset=["Churn"])
    print(f"Dropped {before - len(df)} rows due to invalid target values")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Feature engineering
    service_cols = [
        'PhoneService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies'
    ]
    for c in service_cols:
        if c in df.columns:
            df[c] = df[c].map({"Yes": 1, "No": 0})

    df["total_services"] = df[service_cols].fillna(0).sum(axis=1)
    df["avg_charge_per_service"] = df["MonthlyCharges"] / df["total_services"].replace(0, 1)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1,6,12,24,48,100],
        labels=["0-6","7-12","13-24","25-48","49+"]
    )

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = build_preprocessor(cat_cols, num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    X_train_dense = X_train_prep.toarray() if sparse.issparse(X_train_prep) else X_train_prep
    X_test_dense = X_test_prep.toarray() if sparse.issparse(X_test_prep) else X_test_prep

    results = {}
    models = {}

    # PARAM GRIDS
    grids = {
        "LR": {
            "C": np.logspace(-3, 3, 20)
        },
        "RF": {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "SVM": {
            "C": np.logspace(-2, 2, 10),
            "gamma": ["scale", "auto"]
        },
        "AdaBoost": {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1, 1.0]
        },
        "XGBoost": {
            "n_estimators": [200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    }

    # MODEL TRAINING + TUNING
    experiments = {
        "LR": (LogisticRegression(max_iter=2000), X_train_prep, X_test_prep),
        "RF": (RandomForestClassifier(random_state=42), X_train_dense, X_test_dense),
        "SVM": (SVC(probability=True), X_train_prep, X_test_prep),
        "AdaBoost": (AdaBoostClassifier(random_state=42), X_train_dense, X_test_dense),
        "XGBoost": (xgb.XGBClassifier(eval_metric="logloss"), X_train_dense, X_test_dense)
    }

    for name, (model, Xtr, Xte) in experiments.items():
        print(f"Tuning {name}...")
        with mlflow.start_run(run_name=f"{name}_Tuned"):

            best_model, best_params, cv_auc = tune_model(
                model, grids[name], Xtr, y_train
            )

            metrics = evaluate(best_model, Xte, y_test)

            mlflow.log_param("model", name)
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_roc_auc", cv_auc)
            mlflow.log_metric("test_roc_auc", metrics["roc_auc"])

            results[name] = metrics
            models[name] = best_model

    # SUMMARY
    summary = pd.DataFrame([
        {
            "model": k,
            **v
        } for k, v in results.items()
    ]).sort_values(by="roc_auc", ascending=False)

    print("\n=== FINAL MODEL PERFORMANCE (TUNED) ===")
    print(summary.to_string(index=False))

    best_name = summary.iloc[0]["model"]
    print(f"\nBEST MODEL → {best_name}")

    artifact = {
        "model": models[best_name],
        "preprocessor": preprocessor,
        "model_name": best_name
    }

    joblib.dump(artifact, MODEL_DIR / f"best_model_{best_name}.pkl")
    summary.to_csv(MODEL_DIR / "model_summary_tuned.csv", index=False)

    print("\nArtifacts saved. Pipeline complete.")

if __name__ == "__main__":
    main()

