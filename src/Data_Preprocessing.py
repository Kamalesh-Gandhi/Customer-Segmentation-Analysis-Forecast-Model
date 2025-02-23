import pandas as pd
import logging 
import joblib
from sklearn.preprocessing import LabelEncoder
from utils.helper_functions import Detect_outliers_Zscore


def Label_Encoding(df:pd.DataFrame)-> pd.DataFrame:
    """
    Perform encoding to the Categorical Columns

    Args:
    df --> Pandas DataFrame

    return:
    Pandas DataFrame
    
    """
    try:
        logging.info('Encoding Process Started...')

        category_cols = df.select_dtypes(include=['object','category']).columns.to_list()

        encoders = {}

        for col in category_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder  # Store encoder

        joblib.dump(encoders, "models/Encoder.pkl")   

        return df

        logging.info('Encoding Process Finished')

    except Exception as e:
        logging.error(f'Error in Encoding Process: {e} ')
        raise e


def DataPreProcessing(df:pd.DataFrame) -> pd.DataFrame :

    """
    
    
    """
 
    try:
        
        # Deleting unnecessary columns
        df = df.drop(columns = ['year'],axis = 1)

        #handling missing values
        cols = df.columns.to_list()
        for col in cols:
            if df[col].isna().sum() == 0:
                pass
            else:
                df[col].fillna(df[col].median(),inplace = True)

        # Handling Duplicates Values
        value = df.duplicated().sum()
        if value == 0 :
            pass
        else:
            df.drop_duplicates( inplace = True)

        #Detect Technical Correct Outliers 
        continuos_cols = df.select_dtypes(include=['int','float']).columns.to_list()
        Outliers_data = Detect_outliers_Zscore(df,continuos_cols)

        Encoded_df = Label_Encoding(df)

        return Encoded_df
    
    except Exception as e:
        logging.error(f"Error in Detecting Outliers : {e}")
        raise e


        
        

