import pandas as pd
import joblib
from zenml import step
import mlflow
import logging
from src.Model_Evaluation import (
    evaluate_classification_models, evaluate_regression_models, evaluate_clustering_models
)
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

"""
 Classification Model Evaluation Step
"""
@step(experiment_tracker= experiment_tracker.name)
def evaluate_classification_step(df_test: pd.DataFrame, target_col: str, models_dir: str, depend :dict) -> tuple[dict, dict] :
    """Evaluates trained classification models and logs results to MLFlow."""

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    #  Try loading the scaler, else raise error
    try:
        scaler = joblib.load("models/Classification_scaler.pkl")
        X_test_scaled = scaler.transform(X_test)
    except FileNotFoundError:
        raise ValueError("Classification scaler not found! Ensure it's saved during training.")

    results = evaluate_classification_models(models_dir, X_test_scaled, y_test)

    #  Set MLflow experiment once
    mlflow.set_experiment("classification_evaluation")

    try:
        for model_name, metrics in results.items():

            # 🚨 Ensure no active MLflow runs
            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name=model_name):
                mlflow.log_metrics(metrics)  #  Log performance metrics
            mlflow.end_run()
                

        return results, depend
   
    except Exception as e:
        logging.error(f"Error in Evaluationg Model forClassification {e}")

        # **Ensure MLflow run is ended in case of failure**
        if mlflow.active_run():
            mlflow.end_run()

        raise e


"""
📌 Regression Model Evaluation Step
"""
@step(experiment_tracker=experiment_tracker.name)
def evaluate_regression_step(df_test: pd.DataFrame, target_col: str, models_dir: str, depend: dict) -> tuple[dict, dict]:
    """Evaluates trained regression models and logs results to MLflow."""

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # 🔹 Try loading the scaler, else raise an error
    try:
        scaler = joblib.load("models/Regression_scaler.pkl")
        X_test_scaled = scaler.transform(X_test)
    except FileNotFoundError:
        raise ValueError("🚨 Regression scaler not found! Ensure it's saved during training.")

    # Evaluate regression models
    results = evaluate_regression_models(models_dir, X_test_scaled, y_test)

    # Set MLflow experiment
    mlflow.set_experiment("regression_evaluation")

    try:
        for model_name, metrics in results.items():

            # 🚨 Ensure no active MLflow runs
            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name=model_name):
                mlflow.log_metrics(metrics)  #  Log performance metrics
            mlflow.end_run()

        return results, depend
   
    except Exception as e:
        logging.error(f"Error in Evaluationg Model for Regression {e}")

        # **Ensure MLflow run is ended in case of failure**
        if mlflow.active_run():
            mlflow.end_run()

        raise e
    

"""
📌 Clustering Model Evaluation Step
"""
@step(experiment_tracker=experiment_tracker.name)
def evaluate_clustering_step(df_train:pd.DataFrame, models_dir: str, depend: dict) -> tuple[dict, dict]:
    """Evaluates trained clustering models and logs results to MLflow."""

    # Load scaler for standardization
    try:
        scaler = joblib.load("models/clustering_scaler.pkl")
        X_test_scaled = scaler.transform(df_train)
    except FileNotFoundError:
        raise ValueError("🚨 Clustering scaler not found! Ensure it's saved during training.")

    # Evaluate clustering models
    results = evaluate_clustering_models(models_dir, X_test_scaled)

    # Set MLflow experiment
    mlflow.set_experiment("clustering_evaluation")

    try:
        # Ensure no active MLflow runs before logging
        if mlflow.active_run():
            mlflow.end_run()

        for model_name, metrics in results.items():
            with mlflow.start_run(run_name=f"{model_name}_Clustering"):
                mlflow.log_metrics(metrics)  # Log clustering performance metrics
                
            mlflow.end_run()

        return results, depend

    except Exception as e:
        logging.error(f"🚨 Error in Clustering Model Evaluation: {e}")

        # Ensure MLflow run is ended in case of failure
        if mlflow.active_run():
            mlflow.end_run()

        raise e