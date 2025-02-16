import pandas as pd
import logging 
from utils.helper_functions import Detect_outliers_Zscore


def DataPreProcessing(df:pd.DataFrame) -> pd.DataFrame :
 
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

        return df
    
    except Exception as e:
        logging.error(f"Error in Detecting Outliers : {e}")
        raise e


        
        

