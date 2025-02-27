import pandas as pd
import logging
from zenml import step
from src.Data_Preprocessing import DataPreProcessing
from src.Feature_Engineering import feature_engineering
from utils.helper_functions import Store_ProcessedData


@step
def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Zenml step for Data cleaning ,Feature Engineering 
    
    """
    try:
        logging.info('Data Cleaning Process Started......')
        Processed_data = DataPreProcessing(df)

        if Processed_data is None:
            raise ValueError("🚨 Processed Data is empty. Check the DataPreProcessing process.")

        logging.info("feature Engineering Process Started......")
        FeaturedEngineered_data = feature_engineering(Processed_data)

        if FeaturedEngineered_data is None:
            raise ValueError("🚨 FeatureEngineered data is empty. Check the Feature Engineering process.")

        logging.info("Storing Processed Data to Local System")
        Store_ProcessedData(FeaturedEngineered_data)

        logging.info('Data Cleaning Completed')
        
        return FeaturedEngineered_data

    except Exception as e:
        logging.error(f"Error in Processed data: {e}")
        raise e