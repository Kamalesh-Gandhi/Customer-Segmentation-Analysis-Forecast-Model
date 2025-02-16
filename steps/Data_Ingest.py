import pandas as pd
from zenml import step

@step
def ingest_data(Filepath : str) -> pd.DataFrame:
    """
    ZenML step to ingest data from a given file path.
    
    """
    try:
        df = pd.read_csv(Filepath)
        return df
    except Exception as e:
        print(f'Error in ingest_data : {e}')
        raise e
    