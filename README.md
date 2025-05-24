# Machine Predictive Maintenance Classification

## Overview

This project focuses on building a machine learning model to classify the maintenance needs of industrial machines based on sensor data. The goal is to predict whether a machine requires maintenance, is in good condition, or is at risk of failure. The project uses Python, Docker, and MLflow for model development, tracking, and deployment.

## Project Contents

The project contains the following files and folders:

- **`Dockerfile`**: Defines the Docker image for the project, including the Python environment and dependencies.
- **`requirements.txt`**: Lists the Python packages required for the project.
- **`mlflow_artifacts/`**: Directory to store MLflow artifacts such as models and experiment results.
- **`app.py`**: The main Streamlit application for visualizing predictions and interacting with the model.
- **`data/`**: Contains the dataset used for training and testing the model.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA) and model development.
- **`src/`**: Source code for the project, including data preprocessing, feature engineering, and model training scripts.

## Features

- **Data Preprocessing**: Cleans and prepares raw sensor data for model training.
- **Feature Engineering**: Extracts meaningful features from sensor data.
- **Model Training**: Trains a classification model to predict maintenance needs.
- **Model Tracking**: Uses MLflow to track experiments and manage model versions.
- **Interactive Dashboard**: A Streamlit app for visualizing predictions and interacting with the model.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker
- MLflow
- Streamlit

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/snehaapratap/machine-predictive-maintenance-classification.git
   cd machine-predictive-maintenance-classification
   ```
