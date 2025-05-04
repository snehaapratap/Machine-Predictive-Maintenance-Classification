from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(dag_id='predictive_maintenance_pipeline',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:

    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python scripts/preprocess.py'
    )

    train = BashOperator(
        task_id='train',
        bash_command='python scripts/train.py'
    )

    evaluate = BashOperator(
        task_id='evaluate',
        bash_command='python scripts/evaluate.py'
    )

    preprocess >> train >> evaluate
