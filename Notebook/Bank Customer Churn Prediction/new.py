import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved best model and scaler
model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler_churn.pkl')

# Title of the app
st.title("Bank Customer Churn Prediction")

# Sidebar inputs for customer data
st.sidebar.header("Input Customer Data")
CreditScore = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, step=1)
Geography = st.sidebar.selectbox("Geography", ["Chennai", "Hyderabad", "Other"])
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
Tenure = st.sidebar.number_input("Tenure (years)", min_value=0, max_value=10, step=1)
Balance = st.sidebar.number_input("Balance", min_value=0.0, step=1000.0)
NumOfProducts = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, step=1)
HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=1000.0)

# Submit button inside the sidebar
if st.sidebar.button("Submit"):
    # Create a DataFrame for input data
    input_dict = {
        'CreditScore': CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary
    }

    input_data = pd.DataFrame([input_dict])

    # One-hot encoding for Geography
    input_data['Geography_Chennai'] = (input_data['Geography'] == 'Chennai').astype(int)
    input_data['Geography_Hyderabad'] = (input_data['Geography'] == 'Hyderabad').astype(int)
    input_data = input_data.drop(columns=['Geography'])

    # One-hot encoding for Gender
    input_data['Gender_Male'] = (input_data['Gender'] == 'Male').astype(int)
    input_data = input_data.drop(columns=['Gender'])

    # Align the input features with the model's expected features
    # Extract the expected feature names from the scaler to ensure correct order
    expected_features = scaler.feature_names_in_

    # Reorder input_data to match the expected feature order
    input_data = input_data.reindex(columns=expected_features)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    # Output the result in the main app area
    if prediction[0] == 1:
        st.write("#### Prediction: The customer is likely to churn.")
    else:
        st.write("#### Prediction: The customer is unlikely to churn.")

    # Show prediction probability
    st.write(f"#### Probability of Churn: {prediction_proba[0][1] * 100:.2f}%")
