import pandas as pd
import joblib
from sklearn.metrics import classification_report
import mlflow

def evaluate():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    model = joblib.load("models/model.pkl")

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    
    mlflow.log_metrics({f"f1_class_{k}": v["f1-score"] for k, v in report.items() if isinstance(v, dict)})

if __name__ == "__main__":
    evaluate()
