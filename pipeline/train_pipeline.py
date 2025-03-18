from zenml import pipeline
import logging
import mlflow
import pandas as pd
from steps.ingest_data import ingestdata
from steps.Data_Clean import clean_data
from steps.Train_model import train_classification_models ,train_regression_models, train_clustering_models
from steps.Evaluation_Model import evaluate_classification_step, evaluate_regression_step, evaluate_clustering_step
from steps.feature_selection import feature_selection_classification ,feature_selection_Regression
from steps.best_model import select_best_classification_step, select_best_regression_step, select_best_clustering_step
from utils.helper_functions import Label_Encoding, Store_ProcessedData
from zenml.client import Client
import subprocess

# client = Client()
# # Check if the active stack's experiment tracker is available
# if client.active_stack.experiment_tracker is None:
#     print("Active experiment tracker not found. Initializing ZenML...")
#     # Run ZenML initialization commands (this may vary based on your setup)
#     subprocess.run(["zenml", "init"], check=True)
#     # Optionally, register and set a default stack if needed
#     subprocess.run(["zenml", "stack", "register", "custom_stack", 
#                     "--orchestrator=default", 
#                     "--experiment-tracker=mlflow_tracker", 
#                     "--artifact-store=default"], check=True)
#     subprocess.run(["zenml", "stack", "set", "active", "custom_stack"], check=True)


# ✅ Set MLflow tracking URI & experiment globally
mlflow.set_tracking_uri("file:///D:/DataScience/GUVI/DataScience_GUVI_Projects/Customer_Conversion_Analysis_MLOps/mlruns")
mlflow.set_experiment("customer_conversion_experiment")


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

    #step 1: Loading Data for both Train and Test DataFrame
 
    logging.info(f"starting to fetch the data from the path : {train_data_path}")

    df_train = ingestdata(train_data_path)
    df_test = ingestdata(test_data_path)

    if df_train is None:
        raise ValueError("🚨 df_train is empty! Something went wrong in ingestion.")
    

    #step 2: Cleaning ,Feature Engineering Process 
    logging.info("Started the Cleaning and Feature Engineering Process")

    processed_data_train, processed_data_test = clean_data(df_train, df_test)
    

    if processed_data_train is None:
        raise ValueError("🚨 processed_data_train is empty! Something went wrong in clean Data.")
    
    if processed_data_test is None:
        raise ValueError("🚨 processed_data_test is empty! Something went wrong in clean Data.")
    

    """
     Classification Model Training & Evaluation
    """
    #  Step 3: Feature Selection 
    Classification_df_Train , Classification_df_Test , dependency = feature_selection_classification(processed_data_train, processed_data_test,continuous_cols, categorical_cols, targetcol_classification)

    # Step 4: Training Classification Model 
    try:
        logging.info("Training Model Started for Classification Problem....")

        dependency2 = train_classification_models(Classification_df_Train, targetcol_classification, dependency)

    except Exception as e:
        logging.error(f"Error in Model Training for Classification: {e}")

    # Step 5: Evaluation Classification Model 
    try:
        logging.info("Evaluating Model Started for Classification ....")
        Classification_results, dependency3 = evaluate_classification_step(Classification_df_Test, targetcol_classification, model_dir, dependency2)
    except Exception as e:
        logging.error(f'Error in Evaluating Model for Classification: {e}')

    # Step 6: Finding Best Classification Model 
    try:
        best_classification_model, best_classification_path,dependency4 = select_best_classification_step(Classification_results, model_dir,dependency3)
    except Exception as e:
        logging.error(f'Error in Best Model for Classifcation: {e}')

    """
    Regression Model Training & Evaluation
    """

     #  Step 3: Feature Selection 
    Regression_df_Train , Regression_df_Test, dependency_reg  =feature_selection_Regression(processed_data_train, processed_data_test,continuous_cols, categorical_cols, targetcol_regression, dependency4)
    
    # Step 4: Training Regression Model 
    try:
        logging.info("Training Model Started for Regression Problem....")

        dependency_reg2 = train_regression_models(Regression_df_Train, targetcol_regression, dependency_reg)

    except Exception as e:
        logging.error(f"Error in Model Training for Regression Problem: {e}")

    # Step 5: Evaluation Regression Model 
    try:
        logging.info("Evaluating Model Started for Regression ....")
        Regression_results , dependency_reg3  = evaluate_regression_step(Regression_df_Test, targetcol_regression, model_dir, dependency_reg2)

    except Exception as e:
        logging.error(f'Error in Evaluating Model for Regression: {e}')

    # Step 6: Finding Best Regression Model 
    try:
        best_regression_model, best_regression_path, dependency_reg4 = select_best_regression_step(Regression_results, model_dir, dependency_reg3)
    except Exception as e:
        logging.error(f'Error in Best Model for Classifcation: {e}')


    """
    Clustering Model Training & Evaluation
    """

    # step 4: Training Cluster Model
    try:
        logging.info("Training Model Started for Clustering Problem.....")

        dependency_clust = train_clustering_models(processed_data_train, dependency_reg4, num_clusters = 3 )

    except Exception as e:
        logging.error(f"Error in Model Training for Clustering Problem: {e}")

    # Step 5: Evaluation Clustering Model 
    try:
        logging.info("Evaluating Model Started for Clustering ....")
        clustering_results, dependency_clust2 = evaluate_clustering_step(processed_data_train, model_dir, dependency_clust)
         
    except Exception as e:
        logging.error(f'Error in Evaluating Model for Clustering : {e}')

    # Step 6: Finding Best Classification Model 
    try:
        best_clustering_model, best_clustering_path = select_best_clustering_step(clustering_results , model_dir, dependency_clust2)
    except Exception as e:
        logging.error(f'Error in Best Model for Classifcation: {e}')


    # """
    # ✅ Log Best Models
    # """
    # logging.info(f"🏆 Best Classification Model: {best_classification_model}")
    # logging.info(f"🏆 Best Regression Model: {best_regression_model}")
    # logging.info(f"🏆 Best Clustering Model: {best_clustering_model}")

