import pandas as pd
import os
import joblib
import logging

"""
🔹 Updating the Dataframe using Clustering Model 
"""  
def Update_Dataframe(Train: pd.DataFrame, Test: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame] :
    try:
        logging.info(Train.columns)
        logging.info(Test.columns)
        # Get the absolute path of the project root (one level up from utils)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # Construct the model file path
        model_path = os.path.join(project_root, "models", "K-Means_clustering.pkl")
        logging.info(model_path)
       
       # Construct the model file path
        scaler_path = os.path.join(project_root, "models", "clustering_scaler.pkl")
        scaler = joblib.load(scaler_path)
        scaled_Train = scaler.transform(Train)
        scaled_Test = scaler.transform(Test)


        clustermodel = joblib.load(model_path)
        Train["Customer Segment"] = clustermodel.predict(scaled_Train)
        Test["Customer Segment"] = clustermodel.predict(scaled_Test)

        return Train, Test

    except Exception as e:
        logging.error(f"❌ Error in update Dataframe: {e}")