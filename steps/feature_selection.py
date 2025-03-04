import pandas as pd
import logging
from zenml import step
from src.Feature_Selection import Feature_Selection_Classification
from src.Feature_Selection import Feature_Selection_Regression

@step
def feature_selection_classification(df_train:pd.DataFrame,df_test:pd.DataFrame,cat_cols: list, con_cols:list, target_col: str) -> tuple[pd.DataFrame,pd.DataFrame,dict] :

    """
     Zenml step for feature selection
    
    """
    dependencyholder = {} # Dummy variable to create dependency for next step ,to avoid error in flow

    try:
        logging.info("Feature Selection Process Started for Classification.....")
        Classification_dfTrain, Classification_dfTest = Feature_Selection_Classification(df_train, df_test,cat_cols, con_cols, target_col)
        logging.info("Feature Selection Process Completed for Classification")

        return Classification_dfTrain, Classification_dfTest, dependencyholder

    except Exception as e:
        logging.error("Error in Feature Selection Process for Classication")


@step
def feature_selection_Regression(df_train:pd.DataFrame,df_test:pd.DataFrame,cat_cols: list, con_cols:list, target_col: str, depend:dict ) -> tuple[pd.DataFrame,pd.DataFrame, dict] :

    """
     Zenml step for feature selection for Regression
    
    """
    dependencyholder = {} # Dummy variable to create dependency for next step ,to avoid error in flow

    try:
        logging.info("Feature Selection Process Started for Regression.....")
        Regression_dfTrain, Regression_dfTest = Feature_Selection_Regression(df_train, df_test,cat_cols, con_cols, target_col)
        logging.info("Feature Selection Process Completed for Regression")
        
        return Regression_dfTrain, Regression_dfTest, dependencyholder

    except Exception as e:
        logging.error("Error in Feature Selection Process for Regression")