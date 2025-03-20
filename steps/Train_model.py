import pandas as pd
from zenml import step
import joblib
import mlflow
import logging
from utils.helper_functions import apply_smote
from sklearn.preprocessing import StandardScaler
from src.Model_Training import (train_Linear_regression, train_RandomForest_regressor, train_XGBoost_regressor,
                                train_logistic_regression, train_RandomForest_classifier, train_XGBoost_classifier,
                                train_KMeans, train_dbscan, train_heirarchical)
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


"""
 Classification Models
"""
@step(experiment_tracker=experiment_tracker.name)
def train_classification_models(df: pd.DataFrame, target_col: str, depend: dict) -> dict:
    """Trains classification models and saves them with MLflow logging."""
    
    logging.info(" Starting Classification Model Training...")

    if target_col not in df.columns:
        logging.error(f" Target column '{target_col}' not found in dataset!")
        raise ValueError(f"Target column '{target_col}' not found.")

    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]

    #  Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    #  Save the scaler
    joblib.dump(scaler, "models/Classification_scaler.pkl")
    logging.info(" Saved Classification Scaler: models/Classification_scaler.pkl")

    #  Apply SMOTE for oversampling4
    try:
        X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)
    except Exception as e:
        logging.error(f"Error in implementing SMOTE {e}")
        raise e
    
    models = {
        "Logistic Regression": train_logistic_regression(X_train_resampled, y_train_resampled),
        "Random Forest": train_RandomForest_classifier(X_train_resampled, y_train_resampled),
        "XGBoost": train_XGBoost_classifier(X_train_resampled, y_train_resampled)
    }

    #  Log models with MLflow
    try:
        for model_name, model in models.items():

            # **Ensure the previous run is ended**
            if mlflow.active_run():
                logging.warning("⚠️ An MLflow run was already active. Ending it now...")
                mlflow.end_run()

            with mlflow.start_run(run_name=model_name):
                logging.info(f" Training Classification Model: {model_name}")
                mlflow.log_params(model.get_params())
                model_path = f"models/{model_name.replace(' ', '_')}_classifier.pkl"
                joblib.dump(model, model_path)
                logging.info(f" Model Saved: {model_path}")
                mlflow.log_artifact(model_path)  #  Log model file in MLflow

                # **Explicitly end the run after training**
                mlflow.end_run()

        logging.info(" Classification Model Training Completed!")

        return depend

    except Exception as e:
        logging.error(f"Error in Model Training {e}")

        # **Ensure MLflow run is ended in case of failure**
        if mlflow.active_run():
            mlflow.end_run()

        raise e


"""
 Regression Models
"""
@step(experiment_tracker= experiment_tracker.name)
def train_regression_models(df: pd.DataFrame, target_col: str, depend: dict) -> dict:
    """Trains regression models and saves them."""

    logging.info(" Starting Regression Model Training...")

    if target_col not in df.columns:
        logging.error(f" Target column '{target_col}' not found in dataset!")
        raise ValueError(f"Target column '{target_col}' not found.")

    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]

    #  Standardizing the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    #  Save the scaler
    joblib.dump(scaler, "models/Regression_scaler.pkl")
    logging.info(" Saved Regression Scaler: models/Regression_scaler.pkl")

    models = {
        "Linear Regression": train_Linear_regression(X_train_scaled, y_train),
        "Random Forest": train_RandomForest_regressor(X_train_scaled, y_train),
        "XGBoost": train_XGBoost_regressor(X_train_scaled, y_train)
    }

    try:
        for model_name, model in models.items():

            # **Ensure the previous run is ended**
            if mlflow.active_run():
                logging.warning("⚠️ An MLflow run was already active. Ending it now...")
                mlflow.end_run()

            with mlflow.start_run(run_name=model_name):    
                logging.info(f" Training Regression Model: {model_name}")
                mlflow.log_params(model.get_params())
                model_path = f"models/{model_name.replace(' ', '_')}_regressor.pkl"
                joblib.dump(model, model_path)
                logging.info(f" Model Saved: {model_path}")
                mlflow.log_artifact(model_path)  #  Log model file in MLflow

                # **Explicitly end the run after training**
                mlflow.end_run()

        logging.info(" Regression Model Training Completed!")

        return depend

    except Exception as e:
        logging.error(f"Error in Model Training for Regression Model {e}")

        # **Ensure MLflow run is ended in case of failure**
        if mlflow.active_run():
            mlflow.end_run()

        raise e

"""
Clustering Models Training
"""
@step(experiment_tracker=experiment_tracker.name)
def train_clustering_models(df_train: pd.DataFrame,  num_clusters: int) -> dict:
    """Trains clustering models, logs them with MLflow, and saves the models."""
    
    depend = {} # Dummy variable to create dependency for next step ,to avoid error in flow


    logging.info(" Starting Clustering Model Training...")

    # Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_train)

    # Save the scaler
    joblib.dump(scaler, "models/clustering_scaler.pkl")
    logging.info("Saved Clustering Scaler: models/clustering_scaler.pkl")

    # ✅ **Reduce data size before Hierarchical Clustering**
    SAMPLE_SIZE = min(5000, len(X_scaled))  # Use max 5000 samples or full dataset if smaller
    X_sample = X_scaled[:SAMPLE_SIZE]  # Take a subset

    models = {
        "K-Means": train_KMeans(X_scaled, num_clusters),
        "DBSCAN": train_dbscan(X_scaled),
        "Hierarchical": train_heirarchical(X_sample, num_clusters)
    }

    try:
        # ✅ Ensure no active MLflow runs before logging
        if mlflow.active_run():
            logging.warning("⚠️ An MLflow run was already active. Ending it now...")
            mlflow.end_run()

        for model_name, model in models.items():
            with mlflow.start_run(run_name=f"{model_name}_Clustering"):
                logging.info(f" Training Clustering Model: {model_name}")

                # Log parameters
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                # Save model
                model_path = f"models/{model_name.replace(' ', '_')}_clustering.pkl"
                joblib.dump(model, model_path)
                logging.info(f"Model Saved: {model_path}")

                # Log model artifact in MLflow
                mlflow.log_artifact(model_path)

                # **Explicitly end the run after training**
                mlflow.end_run()

        logging.info(" Clustering Model Training Completed!")

        return depend  # Return dependency for proper execution order

    except Exception as e:
        logging.error(f"🚨 Error in Clustering Model Training: {e}")

        # Ensure MLflow run is ended in case of failure
        if mlflow.active_run():
            mlflow.end_run()

        raise e  # Re-raise exception to stop execution