from pipeline.train_pipeline import trainpipeline
from pipeline.pipeline import pipeline
# from steps.ingest_data import ingestdata
import os

if __name__ == "__main__":

    print(f"Current Working Directory: {os.getcwd()}")

    # df = ingest_data("data/Train_data.csv")
    # print(df.shape)

    # Run the Pipeline
    trainpipeline("data/Train_data.csv", "data/Test_data.csv")
    # pipeline("data/Train_data.csv")
