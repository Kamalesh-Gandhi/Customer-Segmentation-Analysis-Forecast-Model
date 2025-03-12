import pandas as pd
import logging
import mlflow
from zenml import step
import joblib
from zenml.client import Client
from utils.helper_functions import save_best_model
from src.Best_Model import (
    select_best_classification_model, 
    select_best_regression_model, 
    select_best_clustering_model
)

# ✅ Get the MLflow Experiment Tracker
experiment_tracker = Client().active_stack.experiment_tracker


"""
🔹 Select Best Classification Model Step (With MLflow Logging)
"""
@step(experiment_tracker=experiment_tracker.name)
def select_best_classification_step(results: dict, models_dir: str, depend: dict) -> tuple[str, str, dict]:
    try:
        logging.info("🔍 Selecting Best Classification Model...")

        # Select the best model
        best_model_name, best_model_path = select_best_classification_model(results, models_dir)

        # ✅ Load the best model
        best_model = joblib.load(best_model_path)

        # ✅ Ensure no active MLflow runs before logging
        if mlflow.active_run():
            mlflow.end_run()

        # ✅ Start MLflow run inside the ZenML Step
        with mlflow.start_run(run_name="Best_Classification_Model"):
            mlflow.log_metrics(results[best_model_name])  # Log best model's metrics

            
            # ✅ Save and Log Best Model
            saved_model_path = save_best_model(best_model, best_model_name, "classifier", models_dir)
            mlflow.log_artifact(saved_model_path)  # Log best model artifact

        logging.info(f"✅ Best Classification Model Selected: {best_model_name}")

        return best_model_name, saved_model_path, depend
    except Exception as e:
        logging.error(f"❌ Error selecting best classification model: {e}")
        raise e


"""
🔹 Select Best Regression Model Step (With MLflow Logging)
"""
@step(experiment_tracker=experiment_tracker.name)
def select_best_regression_step(results: dict, models_dir: str, depend: dict) -> tuple[str, str, dict]:
    try:
        logging.info("🔍 Selecting Best Regression Model...")

        # Select the best model
        best_model_name, best_model_path = select_best_regression_model(results, models_dir)

        # ✅ Load the best model
        best_model = joblib.load(best_model_path)

        # ✅ Ensure no active MLflow runs before logging
        if mlflow.active_run():
            mlflow.end_run()

        # ✅ Start MLflow run inside the ZenML Step
        with mlflow.start_run(run_name="Best_Regression_Model"):
            mlflow.log_metrics(results[best_model_name])  # Log best model's metrics

            # ✅ Save and Log Best Model
            saved_model_path = save_best_model(best_model, best_model_name, "regressor", models_dir)
            mlflow.log_artifact(saved_model_path)  # Log best model artifact

        logging.info(f"✅ Best Regression Model Selected: {best_model_name}")

        return best_model_name, saved_model_path, depend
    except Exception as e:
        logging.error(f"❌ Error selecting best regression model: {e}")
        raise e


"""
🔹 Select Best Clustering Model Step (With MLflow Logging)
"""
@step(experiment_tracker=experiment_tracker.name)
def select_best_clustering_step(results: dict, models_dir: str, depend: dict) -> tuple[str, str, dict]:
    try:
        logging.info("🔍 Selecting Best Clustering Model...")

        # Select the best model
        best_model_name, best_model_path = select_best_clustering_model(results, models_dir)

        # ✅ Load the best model
        best_model = joblib.load(best_model_path)

        # ✅ Ensure no active MLflow runs before logging
        if mlflow.active_run():
            mlflow.end_run()

        # ✅ Start MLflow run inside the ZenML Step
        with mlflow.start_run(run_name="Best_Clustering_Model"):
            mlflow.log_metrics(results[best_model_name])  # Log best model's metrics

            # ✅ Save and Log Best Model
            saved_model_path = save_best_model(best_model, best_model_name, "clustering", models_dir)
            mlflow.log_artifact(saved_model_path)  # Log best model artifact

        logging.info(f"✅ Best Clustering Model Selected: {best_model_name}")

        return best_model_name, saved_model_path, depend
    except Exception as e:
        logging.error(f"❌ Error selecting best clustering model: {e}")
        raise e
