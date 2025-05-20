import sys
sys.path.append('/opt/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

from train_model import train_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='ML pipeline for multinomial classification',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False
)

def train_task():
    model, scaler, feature_names = train_model()
    return {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }

train_operator = PythonOperator(
    task_id='train_model',
    python_callable=train_task,
    dag=dag
)

# Define task dependencies
train_operator 