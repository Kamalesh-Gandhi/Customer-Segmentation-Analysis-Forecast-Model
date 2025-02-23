from zenml import pipeline
import logging
import pandas as pd
from steps.ingest_data import ingest_data
from steps.Data_Clean import clean_data
from steps.Train_model import train_model
from steps.Evaluation_Model import evaluate_model

@pipeline
def train_pipeline(data_path:str):
    df = ingest_data(data_path)
    processed_data = clean_data(df)
    try:
        logging.info("Training Model Started....")
        train_model(processed_data)
    except Exception as e:
        logging.error(f"Error in Model Training: {e}")

    try:
        logging.info("Evaluating Model Started....")
        evaluate_model(processed_data)
    except Exception as e:
        logging.error(f'Error in Evaluating Model: {e}')