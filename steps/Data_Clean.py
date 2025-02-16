import pandas as pd
import logging
from zenml import step
from src.Data_Preprocessing import DataPreProcessing
from src.Data_Splitting import Datasplit
from src.Feature_Engineering import feature_engineering
from src.Feature_Selection import feature_selection
from utils.helper_functions import Store_ProcessedData
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_data(df:pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.DataFrame,"y_test"],
]:
    """
    Zenml step for Data cleaning ,Feature Engineering ,Feature Selection, and Data Splitting
    
    """
    
    try:
        Processed_data = DataPreProcessing(df)
        Store_ProcessedData(Processed_data)
        FeaturedEngineered_data = feature_engineering(Processed_data)
        FeaturedSelected_data = feature_selection(FeaturedEngineered_data)
        X_train,X_test,y_train,y_test = Datasplit(FeaturedSelected_data)
        logging.info('Data Cleaning Completed')
        
        return X_train,X_test,y_train,y_test

    except Exception as e:
        logging.error(f"Error in Processed data: {e}")
        raise e
