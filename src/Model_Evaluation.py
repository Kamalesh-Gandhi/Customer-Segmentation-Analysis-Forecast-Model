import joblib
import numpy as np
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,silhouette_score, davies_bouldin_score
)

"""
📌 Classification Model Evaluation
"""
def evaluate_classification_models(models_dir, X_test, y_test):
    """Evaluates classification models and returns performance metrics."""

    # 🔹 Ensure models directory exists
    if not os.path.exists(models_dir):
        raise ValueError(f"❌ Models directory '{models_dir}' does not exist!")

    results = {}

    # 🔹 Define model filenames with readable names
    model_files = {
        "Logistic Regression": "Logistic_Regression_classifier.pkl",
        "Random Forest": "Random_Forest_classifier.pkl",
        "XGBoost": "XGBoost_classifier.pkl"
    }

    for model_name, model_file in model_files.items():
        model_path = os.path.join(models_dir, model_file)

        # 🔹 Load model if available
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: {model_name} model file not found! Skipping...")
            continue

        model = joblib.load(model_path)

        # 🔹 Make predictions
        y_pred = model.predict(X_test)

        # 🔹 Calculate metrics safely
        results[model_name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
            "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=1)
        }

        print(f"✅ {model_name} Classification Evaluation Results: {results[model_name]}")

    return results

"""
📌 Regression Model Evaluation
"""
def evaluate_regression_models(models_dir, X_test, y_test):
    """Evaluates regression models and returns performance metrics."""

    # 🔹 Ensure models directory exists
    if not os.path.exists(models_dir):
        raise ValueError(f"❌ Models directory '{models_dir}' does not exist!")

    results = {}

    # 🔹 Define model filenames with readable names
    model_files = {
        "Linear Regression": "Linear_Regression_regressor.pkl",
        "Random Forest": "Random_Forest_regressor.pkl",
        "XGBoost": "XGBoost_regressor.pkl"
    }

    for model_name, model_file in model_files.items():
        model_path = os.path.join(models_dir, model_file)

        # 🔹 Check if the model file exists
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: {model_name} model file not found! Skipping...")
            continue

        # 🔹 Load model and make predictions
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        # 🔹 Calculate evaluation metrics
        results[model_name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R² Score": r2_score(y_test, y_pred)
        }

        print(f"✅ {model_name} Regression Evaluation Results: {results[model_name]}")

    return results


"""
 Clustering Model Evaluation
"""
def evaluate_clustering_models(models_dir, X_scaled):
    """Evaluates clustering models using Silhouette Score & Davies-Bouldin Index."""

    results = {}
    model_files = ["K-Means_clustering.pkl", "DBSCAN_clustering.pkl", "Hierarchical_clustering.pkl"]

    for model_file in model_files:
        model_path = f"{models_dir}/{model_file}"

        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f" Warning: {model_file} not found! Skipping...")
            continue

        # ✅ Use only the same 5,000 samples for evaluation if the model is Hierarchical
        if "Hierarchical" in model_file:
            SAMPLE_SIZE = min(5000, len(X_scaled))
            X_evaluate = X_scaled[:SAMPLE_SIZE]  # ✅ Take only the first 5000 samples
        else:
            X_evaluate = X_scaled  # ✅ Use the full dataset for K-Means & DBSCAN

        #  Handle different clustering models
        if hasattr(model, "labels_"):
            labels = model.labels_  # DBSCAN, Hierarchical
        else:
            labels = model.predict(X_scaled)  # K-Means

        results[model_file] = {
            "Silhouette Score": silhouette_score(X_evaluate, labels),
            "Davies-Bouldin Index": davies_bouldin_score(X_evaluate, labels)
        }

        print(f"✅ {model_file} Clustering Evaluation Results: {results[model_file]}")


    return results
