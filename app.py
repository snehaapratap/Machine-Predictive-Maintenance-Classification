import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessing objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
le_type = joblib.load("le_type.pkl")
le_failure = joblib.load("le_failure.pkl")

# Load original dataset for visuals
df = pd.read_csv("predictive_maintenance.csv")
df = df.drop(columns=["UDI", "Product ID", "Target"])
df['Type'] = le_type.transform(df['Type'])
df['Failure Type'] = le_failure.transform(df['Failure Type'])
X = df.drop("Failure Type", axis=1)
y = df["Failure Type"]

# Streamlit UI
st.title("🔧 Predictive Maintenance - Failure Classifier")
st.markdown("Predict failure type using trained Random Forest model.")

st.sidebar.header("Input Machine Data")
type_input = st.sidebar.selectbox("Type", le_type.classes_)
air_temp = st.sidebar.slider("Air Temperature (K)", float(df['Air temperature [K]'].min()), float(df['Air temperature [K]'].max()))
process_temp = st.sidebar.slider("Process Temperature (K)", float(df['Process temperature [K]'].min()), float(df['Process temperature [K]'].max()))
speed = st.sidebar.slider("Rotational Speed (rpm)", int(df['Rotational speed [rpm]'].min()), int(df['Rotational speed [rpm]'].max()))
torque = st.sidebar.slider("Torque (Nm)", float(df['Torque [Nm]'].min()), float(df['Torque [Nm]'].max()))
tool_wear = st.sidebar.slider("Tool Wear (min)", int(df['Tool wear [min]'].min()), int(df['Tool wear [min]'].max()))

if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
        'Type': [le_type.transform([type_input])[0]],
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear]
    })
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    label = le_failure.inverse_transform([pred])[0]
    st.success(f"Predicted Failure Type: **{label}**")

# Visuals
st.subheader("📊 Failure Type Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x=le_failure.inverse_transform(y), ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("📈 Feature Importances")
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots()
feat_imp.sort_values().plot(kind='barh', ax=ax2)
st.pyplot(fig2)
