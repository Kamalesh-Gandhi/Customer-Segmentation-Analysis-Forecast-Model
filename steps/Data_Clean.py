import pandas as pd
import logging
from zenml import step
from src.Data_Preprocessing import DataPreProcessing
from src.Feature_Engineering import feature_engineering
from utils.helper_functions import Label_Encoding,Store_ProcessedData



@step
def clean_data(df_train:pd.DataFrame , df_test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zenml step for Data cleaning ,Feature Engineering for Train and Test Data
    
    """
    try:
        logging.info('Data Cleaning Process Started......')
        Processed_data_train , Processed_data_test = DataPreProcessing(df_train, df_test)

        if Processed_data_train is None:
            raise ValueError("🚨 Processed Data train is empty. Check the DataPreProcessing process.")
        
        if Processed_data_test is None:
            raise ValueError("🚨 Processed Data test is empty. Check the DataPreProcessing process.")
        

        logging.info("feature Engineering Process Started......")
        FeaturedEngineered_Traindata , FeaturedEngineered_Testdata = feature_engineering(Processed_data_train , Processed_data_test)

        if FeaturedEngineered_Traindata is None:
            raise ValueError("🚨 FeatureEngineered Train data is empty. Check the Feature Engineering process.")

        if FeaturedEngineered_Testdata is None:
            raise ValueError("🚨 FeatureEngineered Test data is empty. Check the Feature Engineering process.")

        logging.info('Data Cleaning Completed')
        
        encoded_df_Train , encoded_df_Test = Label_Encoding(FeaturedEngineered_Traindata, FeaturedEngineered_Testdata ,'page2_clothing_model')

        Store_ProcessedData(encoded_df_Train, 'Processed_Train_Data')
        Store_ProcessedData(encoded_df_Test, 'Processed_Test_Data')

        return encoded_df_Train ,encoded_df_Test

    except Exception as e:
        logging.error(f"Error in Processed data: {e}")
        raise e