import pandas as pd
import logging 
import joblib
from utils.helper_functions import Detect_outliers_Zscore



def DataPreProcessing(dfTrain:pd.DataFrame, dfTest:pd.DataFrame) -> tuple[pd.DataFrame , pd.DataFrame] :

    """
    
    
    """
 
    try:
        
        # Deleting unnecessary columns
        df_Train = dfTrain.drop(columns = ['year'],axis = 1)
        df_Test = dfTest.drop(columns= ['year'], axis = 1)

        #handling missing values and duplicate values in both Train and Test Data
        for df in [df_Train,df_Test]:

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

        #Detect Technical Correct Outliers for Train Data
        continuos_cols = df.select_dtypes(include=['int','float']).columns.to_list()
        Outliers_data = Detect_outliers_Zscore(df,continuos_cols)

        return df_Train, df_Test

    except Exception as e:
        logging.error(f"Error in Detecting Outliers : {e}")
        raise e


        
        

