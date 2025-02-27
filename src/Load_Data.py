import pandas as pd
import logging


def load_data(path: str) -> pd.DataFrame:    
    """
    Load the train data from the given file path.

    Args:
        path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        logging.info(f'Loading data from the path --> {path}')
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logging.error(f'Error loading data from the path --> {e}')
        logging.exception('Full Exception Traceback:')
        raise e