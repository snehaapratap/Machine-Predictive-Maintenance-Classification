# Machine Learning MLOps Pipeline

This project implements a complete MLOps pipeline for multinomial classification using MLflow, Airflow, Docker, and Streamlit.

## Project Structure

```
.
├── app.py                 # Streamlit application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker services orchestration
└── dags/                 # Airflow DAGs
    └── model_training_dag.py
```

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Git

## Setup and Running

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Start the services using Docker Compose:
```bash
docker-compose up --build
```

This will start:
- MLflow server (http://localhost:5000)
- Airflow webserver (http://localhost:8080)
- Streamlit application (http://localhost:8501)

3. Access the services:
- MLflow UI: http://localhost:5000
- Airflow UI: http://localhost:8080 (default credentials: airflow/airflow)
- Streamlit App: http://localhost:8501

## Components

### MLflow
- Tracks experiments
- Logs metrics and parameters
- Stores models and artifacts

### Airflow
- Orchestrates the training pipeline
- Schedules regular model retraining
- Monitors pipeline execution

### Streamlit
- Provides interactive UI for model visualization
- Allows real-time predictions
- Displays model performance metrics

## Usage

1. Place your training data in the `data/` directory
2. Modify the `train_model.py` script to match your data structure
3. The Airflow DAG will automatically trigger model training
4. Use the Streamlit interface to make predictions and view results

## Development

To modify the model or add new features:

1. Update the model training code in `train_model.py`
2. Modify the Streamlit interface in `app.py`
3. Update the Airflow DAG in `dags/model_training_dag.py`
4. Rebuild and restart the containers:
```bash
docker-compose down
docker-compose up --build
```

