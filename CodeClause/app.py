import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

# Load the pre-trained model
model_filename = 'D:\\Internship\\CodeClause\\CreditCard\\random_forest_model.joblib'
rf_classifier = joblib.load(model_filename)

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path of your dataset
dataset_path = "D:\Internship\CodeClause\CreditCard\creditcard.csv"
X = pd.read_csv(dataset_path)

# Assuming 'X' is your DataFrame with features
feature_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

# Function to make predictions
def predict_fraud(features):
    # Reshape input data to match the model's expectations
    features_reshaped = features.reshape(1, -1)
    prediction = rf_classifier.predict(features_reshaped)
    return prediction[0]

# Streamlit app
def main():
    st.title("Credit Card Fraud Detection App")

    # Collect user input for features
    features = []

    st.sidebar.header("User Input")
    for feature_name in feature_names:
        feature_value = st.sidebar.slider(f"Select {feature_name}", float(X[feature_name].min()), float(X[feature_name].max()))
        features.append(feature_value)

    features = np.array(features)

    # Make prediction
    prediction = predict_fraud(features)

    # Display result
    st.subheader("Prediction")
    if prediction == 1:
        st.warning("Potential Fraud Detected!")
        st.bar_chart({"No Fraud": 1 - prediction, "Fraud": prediction})
    else:
        st.success("No Fraud Detected.")
        st.bar_chart({"No Fraud": 1 - prediction, "Fraud": prediction})

if __name__ == '__main__':
    main()
