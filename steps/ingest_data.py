import pandas as pd
from zenml import step
import logging
from src.Load_Data import load_data

@step
def ingestdata(Filepath : str) -> pd.DataFrame:
    """
    ZenML step to ingest data from a given file path.
    """
    try:
        logging.info(f'Ingesting data from: {Filepath}')
        df = load_data(Filepath)
        
        if df is None:
            raise ValueError("🚨 Loaded DataFrame is empty. Check the file format or path.")

        logging.info(f'Data ingestion successful: {df.shape[0]} rows, {df.shape[1]} columns.')

        return df
    
    except Exception as e:
        print(f'Error in ingest_data : {e}')
        raise e