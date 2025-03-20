import pandas as pd
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
import logging
from sklearn.preprocessing import LabelEncoder
import joblib
import os

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
        df.to_csv(f'data/{FileName}.csv',index = False)
        logging.info(f'Stored {FileName} in the Local')

    except Exception as e:
        logging.error(f'Error in Storing the {FileName}: {e}')
        


def apply_smote(X:pd.DataFrame , y:pd.Series) -> pd.DataFrame :
    """
    

    """
    smote = SMOTE(sampling_strategy="auto", random_state=42)  # Use auto for automatic balancing
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled


def Label_Encoding(df_Train:pd.DataFrame, df_Test: pd.DataFrame, col:str, encode:bool)-> tuple[pd.DataFrame,pd.DataFrame]:
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

        if encode == True:
            df_Train[col] = encoder.fit_transform(df_Train[col])
            # df_Test[col] = encoder.transform(df_Test[col])
            df_Test[col] = df_Test[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
            encoders[col] = encoder  # Store encoder

            joblib.dump(encoders, "models/Encoder.pkl")   
        else:

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

            # Construct the model file path
            model_path = os.path.join(project_root, "models", "Encoder.pkl")
            logging.info(model_path)

            encoders = joblib.load(model_path)
            encoder = encoders.get(col, LabelEncoder())
            df_Train[col] = df_Train[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
            df_Test[col] = df_Test[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)


        logging.info('Encoding Process Finished')
        return df_Train , df_Test



    except Exception as e:
        logging.error(f'Error in Encoding Process: {e} ')
        raise e
    
"""
🔹 Save the Best Model
"""
def save_best_model(best_model, model_name: str, model_type: str, models_dir: str) -> str:
    try:
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"best_{model_name.replace(' ', '_')}_{model_type}.pkl")

        # ✅ Save the model using Joblib
        joblib.dump(best_model, model_path)
        logging.info(f"✅ Best {model_type} Model Saved: {model_path}")

        return model_path
    except Exception as e:
        logging.error(f"❌ Error saving best {model_type} model: {e}")
        raise e

