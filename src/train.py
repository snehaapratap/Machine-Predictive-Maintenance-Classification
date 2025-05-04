import pandas as pd
import joblib
import os
import mlflow
from sklearn.ensemble import RandomForestClassifier

def train():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    mlflow.set_experiment("predictive_maintenance")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(model.get_params())

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train()
