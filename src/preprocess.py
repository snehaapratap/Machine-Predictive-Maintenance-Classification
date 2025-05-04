import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess(path="data/raw/predictive_maintenance.csv"):
    df = pd.read_csv(path)
    df = df.drop(['UDI', 'Product ID', 'Target'], axis=1)
    le_type = LabelEncoder()
    le_failure = LabelEncoder()
    df['Type'] = le_type.fit_transform(df['Type'])
    df['Failure Type'] = le_failure.fit_transform(df['Failure Type'])

    X = df.drop('Failure Type', axis=1)
    y = df['Failure Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    joblib.dump(le_failure, "models/label_encoder.pkl")

if __name__ == "__main__":
    preprocess()
