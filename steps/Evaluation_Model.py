import pandas as pd
import joblib
from zenml import step
import mlflow
from src.Model_Evaluation import (
    evaluate_classification_models, evaluate_regression_models, evaluate_clustering_models
)


"""
 Classification Model Evaluation Step
"""
@step
def evaluate_classification_step(df: pd.DataFrame, target_col: str, models_dir: str):
    """Evaluates trained classification models and logs results to MLFlow."""

    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    #  Try loading the scaler, else raise error
    try:
        scaler = joblib.load("models/Classification_scaler.pkl")
        X_test_scaled = scaler.transform(X_test)
    except FileNotFoundError:
        raise ValueError("Classification scaler not found! Ensure it's saved during training.")

    results = evaluate_classification_models(models_dir, X_test_scaled, y_test)

    #  Set MLflow experiment once
    mlflow.set_experiment("classification_evaluation")
    with mlflow.start_run(run_name=model_name):
        for model_name, metrics in results.items():
            mlflow.log_metrics(metrics)  #  Log performance metrics
            print(f" {model_name} Evaluation Results: {metrics}")

    return results


"""
 Regression Model Evaluation Step
"""
@step
def evaluate_regression_step(df: pd.DataFrame, target_col: str, models_dir: str):
    """Evaluates trained regression models and logs results to MLFlow."""

    X_test = df.drop(columns=[target_col])
    y_test = df[target_col]

    try:
        scaler = joblib.load("models/Regression_scaler.pkl")
        X_test_scaled = scaler.transform(X_test)
    except FileNotFoundError:
        raise ValueError("Regression scaler not found! Ensure it's saved during training.")

    results = evaluate_regression_models(models_dir, X_test_scaled, y_test)

    #  Log regression metrics in MLflow
    mlflow.set_experiment("regression_evaluation")
    with mlflow.start_run(run_name="Regression_Models"):
        for model_name, metrics in results.items():
            mlflow.log_metrics(metrics)
            print(f" {model_name} Regression Evaluation Results: {metrics}")

    return results


"""
 Clustering Model Evaluation Step
"""
@step
def evaluate_clustering_step(df: pd.DataFrame, models_dir: str):
    """Evaluates trained clustering models and logs results to MLFlow."""

    try:
        scaler = joblib.load("models/clustering_scaler.pkl")
        X_test_scaled = scaler.transform(df)
    except FileNotFoundError:
        raise ValueError("Clustering scaler not found! Ensure it's saved during training.")

    results = evaluate_clustering_models(models_dir, X_test_scaled)

    #  Log clustering metrics in MLflow
    mlflow.set_experiment("clustering_evaluation")
    with mlflow.start_run(run_name="Clustering_Models"):
        for model_name, metrics in results.items():
            mlflow.log_metrics(metrics)
            print(f" {model_name} Clustering Evaluation Results: {metrics}")

    return results
