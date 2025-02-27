from zenml import pipeline
import logging
import pandas as pd
from steps.ingest_data import ingestdata

@pipeline(enable_cache=False)
def pipeline(Filepath:str):

    try:
        
        logging.info("Data fetching process has started......")
        df = ingestdata(Filepath)

        if df is None:
            raise ValueError("🚨 df_train is empty, Something went wrong in ingestion.")

        # if not isinstance(df,pd.DataFrame):
        #     raise ValueError("🚨 df_train is not DataFrame, Something went wrong in ingestion.") 
        
    except Exception as e:
        raise e
