from pipeline.train_pipeline import trainpipeline
from zenml.client import Client


if __name__ == "__main__":

    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    # Run the Pipeline
    trainpipeline("data/Train_data.csv", "data/Test_data.csv")

