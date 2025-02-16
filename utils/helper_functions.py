import pandas as pd
from scipy.stats import zscore
import logging

def Detect_outliers_Zscore(data : pd.DataFrame,columns,threshold = 3) -> pd.DataFrame:

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

    try:
        df.to_csv('data/Processed_data.csv')
        logging.info('Stored Processed Data in the Local')

    except Exception as e:
        logging.error(f'Error in Storing the Processed Data: {e}')
        


