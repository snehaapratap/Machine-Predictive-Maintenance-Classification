"""
Multinomial Classification DAG using TaskFlow API
"""

from airflow.decorators import dag, task
from pendulum import datetime
import sys
import os

# Add the root directory to sys.path so we can import train_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import train_model

@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 1,
    },
    tags=["ml", "classification"],
)
def multinomial_classification_pipeline():
    @task()
    def run_training():
        train_model()

    run_training()

multinomial_classification_pipeline() 