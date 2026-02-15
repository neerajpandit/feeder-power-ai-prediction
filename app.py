import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("Electricity Load Forecasting Demo")

st.write("Enter next day details to predict load")

# User Inputs
temperature = st.number_input("Temperature (°C)", value=30.0)
load_yesterday = st.number_input("Yesterday Load", value=2000.0)
load_7day_avg = st.number_input("7 Day Average Load", value=1950.0)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
month = st.slider("Month", 1, 12, 1)
is_weekend = 1 if day_of_week in [5,6] else 0

if st.button("Predict Load"):
    
    input_data = np.array([[temperature,
                            load_yesterday,
                            load_7day_avg,
                            day_of_week,
                            month,
                            is_weekend]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Load: {prediction[0]:.2f}")
