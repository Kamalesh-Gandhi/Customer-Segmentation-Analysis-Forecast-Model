import pandas as pd
import numpy as np
import logging


def target_feature(df: pd.DataFrame, col1: str, col2: str, col3: str, feature_name: str) -> pd.DataFrame:
    """
    Create a new feature by combining three existing features

    Args:
    df --> pandas dataframe
    col1 --> column name 1
    col2 --> column name 2
    col3 --> column name 3
    feature_name --> name of the new feature

    Returns:
    pandas dataframe
    """
    try:
        logging.info(f"Creating a new feature called {feature_name} by combining existing features using modified logic")
        # Using vectorized computation with np.where:
        # - Check that the user reached page 5.
        # - Check that the price indicator equals 1.
        # - Check that the order (click count) is greater than 10.
        df[feature_name] = np.where((df[col1] == 5) & (df[col2] == 1) & (df[col3] > 10), 1, 0)
        return df
    except Exception as e:
        logging.error(f"Error occurred when creating a new feature {feature_name} --> : {e}")
        raise e


def new_feature_1(df: pd.DataFrame, col1: str, feature_name: str) -> pd.DataFrame:
    """
    Create a new feature to analyse the time-base feature based on an existing feature.

    """
    try:
        logging.info(f'Creating a new feature called {feature_name} based on an existing feature')
        df[feature_name] = df[col1].apply(lambda x: 1 if x in [6,7] else 0)
        return df
    except Exception as e:
        logging.error(f'Error occured when creating a new feature {feature_name} --> : {e}')
        raise e

def new_feature_2(df: pd.DataFrame, col1: str, col2: str, transform_type: str,feature_name: str) -> pd.DataFrame:
    """
    Create a new feature to analyse the Session-base feature based on an existing feature.

    Args:
    df --> pandas dataframe
    col1 --> column name
    col2 --> column name
    transform_type --> transformation type

    Returns:
    pandas dataframe

    df['total_clicks'] = df.groupby('session_id')['order'].transform('count')
    df['max_page_reached'] = df.groupby('session_id')['page'].transform('max')

    """
    try:
        logging.info(f'Creating a new feature called {feature_name} based on an existing feature')
        df[feature_name] = df.groupby(col1)[col2].transform(transform_type)
        return df
    except Exception as e:
        logging.error(f'Error occured when creating a new feature {feature_name} --> : {e}')
        raise e
    

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame :
    """
    Creating New Features with the help of the existing Data

    Args:
    df --> Pandas DataFrame

    Return:
    Pandas DataFrame

    """

    try:

        logging.info('Starting Feature Engineering Process.........')

        #Creating the Target Feature for Classification Problem
        df = target_feature(df ,col1= 'page' , col2= 'price' , col3= 'order' , feature_name= 'Purchase Completed' )

        # Apply Time-Based Feature (Using Helper Function)
        df = new_feature_1(df , col1 ='day' , feature_name='is_weekend')

        # Apply Session based feature using the existing columns
        df = new_feature_2(df ,col1= 'session_id' ,col2= 'order' ,transform_type= 'count' ,feature_name= 'total_clicks' )

        # Apply Session based feature using the existing columns
        df = new_feature_2(df ,col1= 'session_id' ,col2= 'page' ,transform_type= 'max' ,feature_name= 'max_page_reached' )

        return df

    except Exception as e:
        logging.error(f'Error occured in the Feature Engineering Process')
        raise e



