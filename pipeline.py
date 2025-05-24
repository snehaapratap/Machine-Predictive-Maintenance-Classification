from prefect import flow, task
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

@task(name="load_data", retries=3, retry_delay_seconds=5)
def load_data(file_path: str) -> pd.DataFrame:
    """Load and prepare the dataset."""
    df = pd.read_csv(file_path)
    df = df.drop(columns=["UDI", "Product ID", "Target"])
    return df

@task(name="validate_data")
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the data for missing values and types."""
    assert not df.isnull().any().any(), "Data contains missing values!"
    # Add more validation logic if needed
    return df

@task(name="feature_engineering")
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features to the dataset (dummy for demo)."""
    if 'Air temperature [K]' in df.columns and 'Process pressure [bar]' in df.columns:
        df['EngineTemp_Pressure'] = df['Air temperature [K]'] * df['Process pressure [bar]']
    else:
        print("Warning: Required columns for feature engineering not found. Available columns:", df.columns.tolist())
    return df

@task(name="feature_selection")
def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Select important features (dummy: select all)."""
    # In a real scenario, you might use feature importance or correlation
    return df

@task(name="preprocess_data")
def preprocess_data(df: pd.DataFrame):
    """Preprocess the data including encoding and scaling."""
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

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'le_type': le_type,
        'le_failure': le_failure
    }

@task(name="train_model")
def train_model(data: dict, params: dict):
    """Train the model and log metrics using MLflow."""
    # Set up MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("predictive_maintenance")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(data['X_train'], data['y_train'])
        
        # Make predictions
        y_pred = model.predict(data['X_test'])
        
        # Calculate metrics
        accuracy = accuracy_score(data['y_test'], y_pred)
        report = classification_report(data['y_test'], y_pred, output_dict=True)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{class_name}_{metric_name}", value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return {
            'model': model,
            'accuracy': accuracy,
            'classification_report': report,
            'y_pred': y_pred
        }

@task(name="postprocess_predictions")
def postprocess_predictions(y_pred):
    """Postprocess predictions (dummy: just return as is)."""
    # Could add thresholding, mapping, etc.
    return y_pred

@task(name="evaluate_model")
def evaluate_model(y_true, y_pred):
    """Evaluate the model with extra metrics and log them."""
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mlflow.log_metric("f1_score_weighted", f1)
    # Save confusion matrix as artifact
    cm_path = "artifacts/confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_path, index=False)
    mlflow.log_artifact(cm_path)
    return {'confusion_matrix': cm, 'f1_score': f1}

@task(name="artifact_versioning")
def artifact_versioning():
    """Simulate artifact versioning (dummy: log timestamp)."""
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    with open("artifacts/version.txt", "w") as f:
        f.write(f"Artifact version: {version}\n")
    mlflow.log_artifact("artifacts/version.txt")
    return version

@task(name="save_artifacts")
def save_artifacts(model_data: dict, preprocessed_data: dict):
    """Save model and preprocessing artifacts."""
    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    # Save artifacts
    joblib.dump(model_data['model'], "artifacts/model.pkl")
    joblib.dump(preprocessed_data['scaler'], "artifacts/scaler.pkl")
    joblib.dump(preprocessed_data['le_type'], "artifacts/le_type.pkl")
    joblib.dump(preprocessed_data['le_failure'], "artifacts/le_failure.pkl")
    
    # Log artifacts to MLflow
    mlflow.log_artifact("artifacts/model.pkl")
    mlflow.log_artifact("artifacts/scaler.pkl")
    mlflow.log_artifact("artifacts/le_type.pkl")
    mlflow.log_artifact("artifacts/le_failure.pkl")

@flow(name="predictive_maintenance_pipeline")
def main():
    """Main pipeline flow."""
    # Define model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
    
    # Execute pipeline steps
    df = load_data("predictive_maintenance.csv")
    df = validate_data(df)
    df = feature_engineering(df)
    df = feature_selection(df)
    preprocessed_data = preprocess_data(df)
    model_data = train_model(preprocessed_data, params)
    y_pred = postprocess_predictions(model_data['y_pred'])
    eval_results = evaluate_model(preprocessed_data['y_test'], y_pred)
    save_artifacts(model_data, preprocessed_data)
    artifact_versioning()
    
    print(f"Pipeline completed successfully! Accuracy: {model_data['accuracy']:.4f}")
    print(f"F1 Score (weighted): {eval_results['f1_score']:.4f}")
    print("You can view the MLflow UI by running: mlflow ui")

if __name__ == "__main__":
    main() 