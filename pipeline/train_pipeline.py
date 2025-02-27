from zenml import pipeline
import logging
import pandas as pd
from steps.ingest_data import ingestdata
from steps.Data_Clean import clean_data
from steps.Train_model import train_model
from steps.Evaluation_Model import evaluate_model
from steps.feature_selection import feature_selection_classification

@pipeline(enable_cache=False)
def trainpipeline(train_data_path:str , test_data_path:str):

    """Orchestrates the entire ML pipeline with ZenML and MLflow."""

    # Identify continuous and categorical columns
    continuous_cols = ['month', 'day', 'order', 'country', 'session_id', 'page1_main_category',
                    'colour', 'location', 'model_photography',
                    'price', 'price_2', 'page','is_weekend', 'total_clicks',
                    'max_page_reached']

    categorical_cols = ['page2_clothing_model','Purchase Completed']
    targetcol_classification = 'Purchase Completed'
    targetcol_regression = 'price'
    model_dir = 'models'

    #step 1: Loading Data
 
    logging.info(f"starting to fetch the data from the path : {train_data_path}")

    df_train = ingestdata(train_data_path)
    df_test = ingestdata(test_data_path)

    if df_train is None:
        raise ValueError("🚨 df_train is empty! Something went wrong in ingestion.")
    

    #step 2: Cleaning ,Feature Engineering Process 
    logging.info("Started the Cleaning and Feature Engineering Process")

    processed_data_train = clean_data(df_train)
    

    if processed_data_train is None:
        raise ValueError("🚨 processed_data_train is empty! Something went wrong in clean Data.")



    """
     Classification Model Training & Evaluation
    """
    #  Step 3: Feature Selection 
    Classification_df_features = feature_selection_classification(processed_data_train, continuous_cols, categorical_cols, targetcol_classification)



    # Classification_df_train = processed_data_train[Classification_df_features]

    # if Classification_df_train is None:
    #     raise ValueError("🚨 Classification_df_train is empty! Something went wrong in Feature Selection Classification.")


    # Classification_testdata = df_test[Classification_columns]  # Use same features as train

    # if Classification_testdata is None:
    #     raise ValueError("🚨 Classification_testdata is empty! Something went wrong in Feature Selection Classification.")

    # logging.info(f'Classification Test Data Shape: {Classification_testdata.shape[0]} rows, {Classification_testdata.shape[1]} columns.')


    # try:
    #     logging.info("Training Model Started....")
    #     # Classification_traindata.head(10)
    #     train_model(Classification_traindata)
    # except Exception as e:
    #     logging.error(f"Error in Model Training: {e}")

    # try:
    #     logging.info("Evaluating Model Started....")
    #     # evaluate_model(processed_data_test)
    # except Exception as e:
    #     logging.error(f'Error in Evaluating Model: {e}')



