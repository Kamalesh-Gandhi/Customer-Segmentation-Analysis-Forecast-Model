import pandas as pd
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import logging
from sklearn.preprocessing import LabelEncoder
import joblib

def Detect_outliers_Zscore(data : pd.DataFrame,columns,threshold = 3) -> pd.DataFrame:

    """
    Detecting the Outliers in the Data by using Zscore method

    Args:
    Data --> Pandas DataFrame
    columns --> apply the zscore to particular columns of Data
    threshold --> Value to segregate the Outliers 

    Return:
    Pandas DataFrame 
    
    """

    try:
        outlier_stats = []
        for col in columns:
            # Calculate Z-scores using scipy's zscore method
            z_scores = zscore(data[col])
            
            # Count outliers based on the threshold
            outliers = (abs(z_scores) > threshold).sum()
            
            # Append results for this column
            outlier_stats.append({
                "Column": col,
                "Outliers": outliers
            })
        logging.info('Detecting Outliers in the data')

        return pd.DataFrame(outlier_stats)
    
    
    except Exception as e:
        logging.error(f"Error in Detecting Outliers : {e}")
        raise e


def Store_ProcessedData(df : pd.DataFrame,FileName) -> pd.DataFrame :
    """
    Store the Processed Data into the Local File

    Args:
    df --> Pandas DataFrame

    Returns:
    Pandas DataFrame

    """

    try:
        df.to_csv(f'data/{FileName}.csv')
        logging.info(f'Stored {FileName} in the Local')

    except Exception as e:
        logging.error(f'Error in Storing the {FileName}: {e}')
        


def apply_smote(X:pd.DataFrame , y:pd.Series) -> pd.DataFrame :
    """
    

    """
    smote = SMOTE(sampling_strategy="auto", random_state=42)  # Use auto for automatic balancing
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled


def Label_Encoding(df_Train:pd.DataFrame, df_Test: pd.DataFrame, col:str)-> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Perform encoding to the Categorical Columns

    Args:
    df --> Pandas DataFrame

    return:
    Pandas DataFrame
    
    """
    try:
        logging.info('Encoding Process Started...')

        encoders = {}
        encoder = LabelEncoder()

        df_Train[col] = encoder.fit_transform(df_Train[col])
        # df_Test[col] = encoder.transform(df_Test[col])
        df_Test[col] = df_Test[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
        encoders[col] = encoder  # Store encoder

        joblib.dump(encoders, "models/Encoder.pkl")   

        logging.info('Encoding Process Finished')
        return df_Train , df_Test



    except Exception as e:
        logging.error(f'Error in Encoding Process: {e} ')
        raise e