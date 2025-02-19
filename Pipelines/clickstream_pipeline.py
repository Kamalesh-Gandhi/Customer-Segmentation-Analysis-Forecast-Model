import pandas as pd
import joblib
from zenml import pipeline
from steps.Data_Ingest import ingest_data
from steps.Data_Clean import clean_data
from steps.Train_Model import train_classification_models, train_clustering_models, train_regression_models
from steps.Evaluation_Model import evaluate_classification_step, evaluate_clustering_step, evaluate_regression_step
from src.Feature_Selection import Feature_Selection_Classification, Feature_Selection_Regression


@pipeline
def clickstream_pipeline(Train_FilePath: str, Test_FilePath: str):
    
    """Orchestrates the entire ML pipeline with ZenML and MLflow."""

    # Identify continuous and categorical columns
    continuous_cols = ['month', 'day', 'order', 'country', 'session_id', 'page1_main_category',
                    'colour', 'location', 'model_photography',
                    'price', 'price_2', 'page','is_weekend', 'total_clicks',
                    'max_page_reached']

    categorical_cols = ['page2_clothing_model','target']
    targetcol_classification = 'Purchase Completed'
    targetcol_regression = 'price'
    model_dir = 'models'

    #  Step 1: Ingest Data
    df_train = ingest_data(Train_FilePath)
    df_test = ingest_data(Test_FilePath)  

    #  Step 2: Clean Data
    ProcessedData_train = clean_data(df_train)
    ProcessedData_test = clean_data(df_test)

    """
     Classification Model Training & Evaluation
    """
    #  Step 3: Feature Selection
    Selected_Classification_Feature_train = Feature_Selection_Classification(ProcessedData_train, continuous_cols, categorical_cols, targetcol_classification)
    if not Selected_Classification_Feature_train:
        raise ValueError("No features were selected for classification.")

    Classification_traindata = ProcessedData_train[Selected_Classification_Feature_train]
    Classification_testdata = ProcessedData_test[Selected_Classification_Feature_train]  # Use same features as train

    #  Step 4: Train Classification Models
    train_classification_models(Classification_traindata, targetcol_classification)
    #  Step 5: Evaluate Classification Models
    Classification_problem_Results = evaluate_classification_step(Classification_testdata, targetcol_classification, model_dir)
    print(pd.DataFrame(Classification_problem_Results))


    """
     Regression Model Training & Evaluation
    """
    #  Step 6: Feature Selection
    Selected_regression_Feature_train = Feature_Selection_Regression(ProcessedData_train, continuous_cols, categorical_cols, targetcol_regression)
    if not Selected_regression_Feature_train:
        raise ValueError("No features were selected for regression.")

    Regression_traindata = ProcessedData_train[Selected_regression_Feature_train]
    Regression_testdata = ProcessedData_test[Selected_regression_Feature_train]  # Use same features as train

    #  Step 7: Train Regression Models
    train_regression_models(Regression_traindata, targetcol_regression)

    #  Step 8: Evaluate Regression Models
    Regression_problem_Results = evaluate_regression_step(Regression_testdata, targetcol_regression, model_dir)
    print(pd.DataFrame(Regression_problem_Results))


    """
     Clustering Model Training & Evaluation
    """
    #  Step 9: Train Clustering Models
    train_clustering_models(Classification_traindata)  # Use standardized data

    #  Step 10: Evaluate Clustering Models
    Clustering_problem_Results = evaluate_clustering_step(ProcessedData_test, model_dir)  # ✅ FIXED Variable Name
    print(pd.DataFrame(Clustering_problem_Results))
