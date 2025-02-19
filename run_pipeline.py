from Pipelines.clickstream_pipeline import clickstream_pipeline

if __name__ == '__main__':
    #Run the Pipeline
    clickstream_pipeline("data\Train_data.csv", "data\Test_data.csv")