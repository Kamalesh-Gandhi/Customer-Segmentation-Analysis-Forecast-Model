from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


default_args = {
                "owner":"airflow",
                "depend_on_past": False,
                "start_date": datetime(2025,3,13),
                "email_for_failure":False,
                "email_for_retry": False,
                "retries": 1,
                "retry_delay":timedelta(minutes=5)
}

with DAG (
    "train_pipeline_ADg_bash",
    default_args=default_args,
    description= "dag to run the train pipeline on the bash operator",
    schedule_interval='@daily',
    catchup= False
) as dag:
    
    run_pipeline = BashOperator(
        task_id = 'run train pipeline',
        bash_command= 'python run_pipeline.py'
    )

