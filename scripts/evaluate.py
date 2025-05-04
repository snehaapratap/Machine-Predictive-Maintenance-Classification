import pandas as pd
import joblib
import mlflow
from sklearn.metrics import classification_report

def evaluate():
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    model = joblib.load("models/model.pkl")
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("predictive_maintenance")

    with mlflow.start_run(nested=True):
        mlflow.log_metrics({f"f1_class_{k}": v["f1-score"] for k, v in report.items() if isinstance(v, dict)})

if __name__ == "__main__":
    evaluate()
