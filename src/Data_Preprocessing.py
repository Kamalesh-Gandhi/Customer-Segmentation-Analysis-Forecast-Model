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
        logging.info('encoding Process Started...')

        category_col = df.select_dtypes(include=['object','category']).columns.to_list()

        encoder = LabelEncoder()
        df[category_col] = encoder.fit_transform(df[category_col])
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
        Outliers_data = Detect_outliers_Zscore(df)

        Encoded_df = Label_Encoding(df)

        joblib.dump(Encoded_df, "models/Encoder.pkl")

        return Encoded_df
    
    except Exception as e:
        logging.error(f"Error in Detecting Outliers : {e}")
        raise e


        
        

