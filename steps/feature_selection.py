import pandas as pd
import logging
from zenml import step
from src.Feature_Selection import Feature_Selection_Classification
from src.Feature_Selection import Feature_Selection_Regression

@step
def feature_selection_classification(df:pd.DataFrame,cat_cols: list, con_cols:list, target_col: str) -> list :

    """
     Zenml step for feature selection
    
    """
    try:
        logging.info("Feature Selection Process Started.....")
        Classification_df_features = Feature_Selection_Classification(df, cat_cols, con_cols, target_col)

        if Classification_df_features is None:
            raise ValueError("🚨 Classification_df_features is empty! Something went wrong in Feature Selection Classification.")

        logging.info(f'Classification Tain Features: {Classification_df_features}')

        Classification_Train_df = df[Classification_df_features]

        logging.info(Classification_Train_df.head())
        
        return Classification_df_features

    except Exception as e:
        logging.error("Error in Feature Selection Process")