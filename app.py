import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

def load_latest_model():
    # Load the latest model from MLflow
    client = mlflow.tracking.MlflowClient()
    latest_run = client.search_runs(experiment_ids=["0"], max_results=1)[0]
    model_uri = f"runs:/{latest_run.info.run_id}/model"
    scaler_uri = f"runs:/{latest_run.info.run_id}/scaler"
    
    model = mlflow.sklearn.load_model(model_uri)
    scaler = mlflow.sklearn.load_model(scaler_uri)
    
    return model, scaler, latest_run

def main():
    st.title("Multinomial Classification Model Dashboard")
    
    # Load model and metrics
    try:
        model, scaler, latest_run = load_latest_model()
        
        # Display model metrics
        st.header("Model Performance Metrics")
        metrics = latest_run.data.metrics
        st.write(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
        
        # Create a DataFrame for metrics visualization
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        # Plot metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=metrics_df, x='Metric', y='Value')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Prediction interface
        st.header("Make Predictions")
        
        # Create input fields for features
        # Note: You'll need to modify this based on your actual features
        feature1 = st.number_input("Feature 1", value=0.0)
        feature2 = st.number_input("Feature 2", value=0.0)
        feature3 = st.number_input("Feature 3", value=0.0)
        
        if st.button("Predict"):
            # Prepare input data
            input_data = np.array([[feature1, feature2, feature3]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            
            st.write("Prediction:", prediction[0])
            st.write("Class Probabilities:")
            
            # Plot probability distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=model.classes_, y=probabilities[0])
            plt.title("Class Probabilities")
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure MLflow server is running and a model has been trained.")

if __name__ == "__main__":
    main() 