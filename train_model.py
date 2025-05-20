import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    # Load your data here - replace with your actual data loading logic
    # This is a placeholder - you'll need to modify this based on your data
    data = pd.read_csv('data/your_data.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def train_model():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # Start MLflow run
    with mlflow.start_run(run_name="multinomial_classification"):
        # Load and prepare data
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{class_name}_{metric_name}", value)
        
        # Create and log confusion matrix plot
        plt = plot_confusion_matrix(y_test, y_pred, model.classes_)
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save scaler
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Save feature names
        feature_names = X.columns.tolist()
        mlflow.log_param("feature_names", feature_names)
        
        print(f"Model accuracy: {accuracy}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, scaler, feature_names

if __name__ == "__main__":
    train_model() 