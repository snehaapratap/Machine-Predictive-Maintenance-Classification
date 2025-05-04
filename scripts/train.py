import pandas as pd
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier

def train():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    mlflow.set_tracking_uri("http://localhost:5000")  # MLflow tracking server
    mlflow.set_experiment("predictive_maintenance")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_params(model.get_params())
        joblib.dump(model, "models/model.pkl")

if __name__ == "__main__":
    train()
