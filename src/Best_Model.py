import logging
import joblib
import pandas as pd
import mlflow
import os

"""
🔹 Select Best Classification Model Based on Accuracy
"""
def select_best_classification_model(results: dict, models_dir: str) -> tuple[str, str]:
    try:
        best_model_name = max(results, key=lambda x: results[x]["Accuracy"])
        best_model_path = os.path.join(models_dir, f"{best_model_name.replace(' ', '_')}_classifier.pkl")

        logging.info(f"✅ Best Classification Model: {best_model_name}")

        return best_model_name, best_model_path
    except Exception as e:
        logging.error(f"❌ Error selecting best classification model: {e}")
        raise e


"""
🔹 Select Best Regression Model Based on R² Score
"""
def select_best_regression_model(results: dict, models_dir: str)-> tuple[str, str]:
    try:
        best_model_name = max(results, key=lambda x: results[x]["R² Score"])
        best_model_path = os.path.join(models_dir, f"{best_model_name.replace(' ', '_')}_regressor.pkl")

        logging.info(f"✅ Best Regression Model: {best_model_name}")

        return best_model_name, best_model_path
    except Exception as e:
        logging.error(f"❌ Error selecting best regression model: {e}")
        raise e


"""
🔹 Select Best Clustering Model Based on Silhouette Score
"""
def select_best_clustering_model(results: dict, models_dir: str)-> tuple[str, str]:
    try:
        best_model_name = max(results, key=lambda x: results[x]["Silhouette Score"])
        best_model_path = os.path.join(models_dir, f"{best_model_name.replace(' ', '_')}_clustering.pkl")

        logging.info(f"✅ Best Clustering Model: {best_model_name}")

        return best_model_name, best_model_path
    except Exception as e:
        logging.error(f"❌ Error selecting best clustering model: {e}")
        raise e
