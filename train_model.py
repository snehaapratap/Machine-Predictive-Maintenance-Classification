import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os

# Set up local MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")

# Create/Set experiment
mlflow.set_experiment("predictive_maintenance")

# Load data
df = pd.read_csv("predictive_maintenance.csv")
df = df.drop(columns=["UDI", "Product ID", "Target"])

# Encode categorical columns
le_type = LabelEncoder()
df['Type'] = le_type.fit_transform(df['Type'])

le_failure = LabelEncoder()
df['Failure Type'] = le_failure.fit_transform(df['Failure Type'])

# Prepare features and target
X = df.drop("Failure Type", axis=1)
y = df["Failure Type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    for class_name, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{class_name}_{metric_name}", value)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Save artifacts
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le_type, "le_type.pkl")
    joblib.dump(le_failure, "le_failure.pkl")
    
    # Log artifacts
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("scaler.pkl")
    mlflow.log_artifact("le_type.pkl")
    mlflow.log_artifact("le_failure.pkl")

print("Training completed! You can view the MLflow UI by running: mlflow ui")
