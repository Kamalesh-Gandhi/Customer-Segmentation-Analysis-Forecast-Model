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
        FeaturedEngineered_data = feature_engineering(Processed_data)
        Store_ProcessedData(FeaturedEngineered_data)
        logging.info('Data Cleaning Completed')
        
        return FeaturedEngineered_data

    except Exception as e:
        logging.error(f"Error in Processed data: {e}")
        raise e