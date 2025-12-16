# src/train.py

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_processing import prepare_model_dataset
from src.target_engineering import build_proxy_target
from src.evaluation import evaluate_model


def load_and_prepare_data(path: str):
    df = pd.read_csv(path)

    # Feature engineering (Task 3)
    processed_df, preprocessor = prepare_model_dataset(df)

    # Target engineering (Task 4)
    target_df = build_proxy_target(df)

    final_df = processed_df.merge(target_df, on="CustomerId", how="left")

    X = preprocessor.fit_transform(final_df)
    y = final_df["is_high_risk"]

    return X, y, preprocessor


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    experiments = []

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    lr = LogisticRegression(max_iter=1000)

    lr_params = {
        "C": [0.01, 0.1, 1.0],
        "solver": ["lbfgs"],
    }

    lr_grid = GridSearchCV(
        lr,
        lr_params,
        scoring="roc_auc",
        cv=3,
    )

    lr_grid.fit(X_train, y_train)

    experiments.append(("LogisticRegression", lr_grid))

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf = RandomForestClassifier(random_state=42)

    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
    }

    rf_grid = GridSearchCV(
        rf,
        rf_params,
        scoring="roc_auc",
        cv=3,
    )

    rf_grid.fit(X_train, y_train)

    experiments.append(("RandomForest", rf_grid))

    # -----------------------------
    # MLflow Tracking
    # -----------------------------
    mlflow.set_experiment("credit-risk-model")

    for name, model in experiments:
        with mlflow.start_run(run_name=name):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_proba)
            mlflow.sklearn.log_model(
                preprocessor, 
                artifact_path="preprocessor",
                registered_model_name=f"{name}_Preprocessor" # Log it separately for simplicity
            )

            mlflow.log_params(model.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model.best_estimator_,
                artifact_path="model",
                registered_model_name="CreditRiskModel",
            )


if __name__ == "__main__":
    X, y, preprocessor = load_and_prepare_data("data/raw/data.csv")
    train_models(X, y)
