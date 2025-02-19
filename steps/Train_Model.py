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


#  Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


"""
 Classification Models
"""
@step
def train_classification_models(df: pd.DataFrame, target_col: str):
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

    #  Apply SMOTE for oversampling
    X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train)

    models = {
        "Logistic Regression": train_logistic_regression(X_train_resampled, y_train_resampled),
        "Random Forest": train_RandomForest_classifier(X_train_resampled, y_train_resampled),
        "XGBoost": train_XGBoost_classifier(X_train_resampled, y_train_resampled)
    }

    #  Log models with MLflow
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logging.info(f" Training Model: {model_name}")
            mlflow.log_params(model.get_params())
            model_path = f"models/{model_name.replace(' ', '_')}_classifier.pkl"
            joblib.dump(model, model_path)
            logging.info(f" Model Saved: {model_path}")
            mlflow.log_artifact(model_path)  #  Log model file in MLflow

    logging.info(" Classification Model Training Completed!")


"""
 Regression Models
"""
@step
def train_regression_models(df: pd.DataFrame, target_col: str):
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

    for model_name, model in models.items():
        model_path = f"models/{model_name.replace(' ', '_')}_regressor.pkl"
        joblib.dump(model, model_path)
        logging.info(f" Model Saved: {model_path}")

    logging.info(" Regression Model Training Completed!")


"""
 Clustering Models
"""
@step
def train_clustering_models(df: pd.DataFrame, num_clusters: int = 3):
    """Trains clustering models and saves them."""

    logging.info(" Starting Clustering Model Training...")

    #  Standardizing the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    #  Save the scaler
    joblib.dump(scaler, "models/clustering_scaler.pkl")
    logging.info(" Saved Clustering Scaler: models/clustering_scaler.pkl")

    models = {
        "K-Means": train_KMeans(X_scaled, num_clusters),
        "DBSCAN": train_dbscan(X_scaled),
        "Hierarchical": train_heirarchical(X_scaled, num_clusters)
    }

    for model_name, model in models.items():
        model_path = f"models/{model_name.replace(' ', '_')}_clustering.pkl"
        joblib.dump(model, model_path)
        logging.info(f" Model Saved: {model_path}")

    logging.info(" Clustering Model Training Completed!")
