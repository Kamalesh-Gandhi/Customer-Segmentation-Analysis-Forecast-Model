import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,silhouette_score, davies_bouldin_score
)


"""
 Classification Model Evaluation
"""
def evaluate_classification_models(models_dir, X_test, y_test):
    """Evaluates classification models and returns performance metrics."""
    
    results = {}
    model_files = ["Logistic_Regression_classifier.pkl", "Random_Forest_classifier.pkl", "XGBoost_classifier.pkl"]

    for model_file in model_files:
        model_path = f"{models_dir}/{model_file}"
        
        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f" Warning: {model_file} not found! Skipping...")
            continue

        y_pred = model.predict(X_test)

        results[model_file] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
            "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=1)
        }

    return results


"""
 Regression Model Evaluation
"""
def evaluate_regression_models(models_dir, X_test, y_test):
    """Evaluates regression models and returns performance metrics."""

    results = {}
    model_files = ["Linear_Regression_regressor.pkl", "Random_Forest_regressor.pkl", "XGBoost_regressor.pkl"]

    for model_file in model_files:
        model_path = f"{models_dir}/{model_file}"

        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            print(f" Warning: {model_file} not found! Skipping...")
            continue

        y_pred = model.predict(X_test)

        results[model_file] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),  
            "R² Score": r2_score(y_test, y_pred)
        }

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

        #  Handle different clustering models
        if hasattr(model, "labels_"):
            labels = model.labels_  # DBSCAN, Hierarchical
        else:
            labels = model.predict(X_scaled)  # K-Means

        results[model_file] = {
            "Silhouette Score": silhouette_score(X_scaled, labels),
            "Davies-Bouldin Index": davies_bouldin_score(X_scaled, labels)
        }

    return results
