import pandas as pd
import os
import joblib
import logging
from src.Update_DataFrame import Update_Dataframe

"""
🔹 Updating the Dataframe using Clustering Model 
"""  
def update_dataframe(Train: pd.DataFrame, Test: pd.DataFrame, depend:dict) -> tuple[pd.DataFrame,pd.DataFrame,dict] :
    try:
        
        updatedTrain , updatedTest = Update_Dataframe(Train, Test)
        return updatedTrain, updatedTest, depend

    except Exception as e:
        logging.error(f"❌ Error in calling update Dataframe: {e}")