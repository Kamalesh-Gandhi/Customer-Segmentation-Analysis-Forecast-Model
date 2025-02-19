import pandas as pd
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import logging

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


def Store_ProcessedData(df : pd.DataFrame) -> pd.DataFrame :
    """
    Store the Processed Data into the Local File

    Args:
    df --> Pandas DataFrame

    Returns:
    Pandas DataFrame

    """

    try:
        df.to_csv('data/Processed_data.csv')
        logging.info('Stored Processed Data in the Local')

    except Exception as e:
        logging.error(f'Error in Storing the Processed Data: {e}')
        


def apply_smote(X:pd.DataFrame , y:pd.Series) -> pd.DataFrame :
    """
    

    """
    smote = SMOTE(sampling_strategy="auto", random_state=42)  # Use auto for automatic balancing
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled