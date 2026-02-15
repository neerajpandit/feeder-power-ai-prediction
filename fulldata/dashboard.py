# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import timedelta

# st.set_page_config(layout="wide")

# # Load model and encoder
# # model = joblib.load("energy_model.pkl")
# # encoder = joblib.load("feeder_encoder.pkl")
# import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model_path = os.path.join(BASE_DIR, "energy_model.pkl")
# encoder_path = os.path.join(BASE_DIR, "feeder_encoder.pkl")

# model = joblib.load(model_path)
# encoder = joblib.load(encoder_path)


# # Load dataset
# # data = pd.read_csv("energy_feeder_dataset.csv")
# data_path = os.path.join(BASE_DIR, "energy_feeder_dataset.csv")
# data = pd.read_csv(data_path)

# data["Date"] = pd.to_datetime(data["Date"])

# # Feature Engineering
# data = data.sort_values(["Feeder_ID","Date"])
# data["Load_yesterday"] = data.groupby("Feeder_ID")["Input_Energy_MWh"].shift(1)
# data["Load_7day_avg"] = data.groupby("Feeder_ID")["Input_Energy_MWh"].rolling(7).mean().reset_index(0,drop=True)
# data["Day_of_week"] = data["Date"].dt.dayofweek
# data["Month"] = data["Date"].dt.month
# data["Is_weekend"] = data["Day_of_week"].isin([5,6]).astype(int)

# data = data.dropna()

# # Sidebar
# st.sidebar.title("⚡ AI Energy Audit Control Panel")

# feeder_list = data["Feeder_ID"].unique()
# selected_feeder = st.sidebar.selectbox("Select Feeder", feeder_list)

# available_supply = st.sidebar.number_input("Available Supply (MWh)", value=1000.0)

# future_temp = st.sidebar.number_input("Tomorrow Temperature (°C)", value=5.0)

# # Filter feeder data
# feeder_data = data[data["Feeder_ID"] == selected_feeder]
# last_row = feeder_data.iloc[-1]

# # Encode feeder
# feeder_code = encoder.transform([selected_feeder])[0]

# # Forecast tomorrow
# input_features = np.array([[
#     future_temp,
#     last_row["Input_Energy_MWh"],
#     last_row["Load_7day_avg"],
#     (last_row["Date"] + timedelta(days=1)).weekday(),
#     (last_row["Date"] + timedelta(days=1)).month,
#     1 if (last_row["Date"] + timedelta(days=1)).weekday() in [5,6] else 0,
#     feeder_code
# ]])

# predicted_load = model.predict(input_features)[0]
# deficit = predicted_load - available_supply

# # Main Dashboard
# st.title("🏛 AI Energy Audit Command Center")

# col1, col2, col3 = st.columns(3)

# col1.metric("🔮 Predicted Load (MWh)", round(predicted_load,2))
# col2.metric("⚡ Available Supply (MWh)", available_supply)
# col3.metric("🚨 Power Deficit (MWh)", round(deficit,2))

# st.subheader("📈 Historical Load Trend")
# st.line_chart(feeder_data.set_index("Date")["Input_Energy_MWh"])

# # Loss Calculation
# st.subheader("📊 Feeder Loss Analysis")

# data["Loss_%"] = ((data["Input_Energy_MWh"] - data["Billed_Energy_MWh"]) / data["Input_Energy_MWh"]) * 100

# loss_rank = data.groupby("Feeder_ID")["Loss_%"].mean().sort_values(ascending=False).reset_index()

# st.dataframe(loss_rank)

# # Anomaly Detection
# threshold = data["Loss_%"].mean() + 2 * data["Loss_%"].std()
# data["Anomaly"] = data["Loss_%"] > threshold

# st.subheader("🚨 Theft / High Loss Alerts")

# anomalies = data[data["Anomaly"] == True][["Date","Feeder_ID","Loss_%"]]

# st.dataframe(anomalies.tail(10))

# st.success("System Running Successfully ✅")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta

st.set_page_config(layout="wide")

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "energy_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "feeder_encoder.pkl"))
data = pd.read_csv(os.path.join(BASE_DIR, "energy_feeder_dataset.csv"))

data["Date"] = pd.to_datetime(data["Date"])

# ---------------- Feature Engineering ----------------
data = data.sort_values(["Feeder_ID","Date"])
data["Load_yesterday"] = data.groupby("Feeder_ID")["Input_Energy_MWh"].shift(1)
data["Load_7day_avg"] = data.groupby("Feeder_ID")["Input_Energy_MWh"].rolling(7).mean().reset_index(0,drop=True)
data["Day_of_week"] = data["Date"].dt.dayofweek
data["Month"] = data["Date"].dt.month
data["Is_weekend"] = data["Day_of_week"].isin([5,6]).astype(int)

data = data.dropna()

# ---------------- Sidebar ----------------
st.sidebar.title("⚡ AI Energy Audit Control Panel")

feeder_list = data["Feeder_ID"].unique()
selected_feeder = st.sidebar.selectbox("Select Feeder", feeder_list)

available_supply = st.sidebar.number_input("Available Supply (MWh)", value=1000.0)
future_temp = st.sidebar.number_input("Tomorrow Temperature (°C)", value=5.0)

# ---------------- Forecast ----------------
feeder_data = data[data["Feeder_ID"] == selected_feeder]
last_row = feeder_data.iloc[-1]

feeder_code = encoder.transform([selected_feeder])[0]

input_features = np.array([[ 
    future_temp,
    last_row["Input_Energy_MWh"],
    last_row["Load_7day_avg"],
    (last_row["Date"] + timedelta(days=1)).weekday(),
    (last_row["Date"] + timedelta(days=1)).month,
    1 if (last_row["Date"] + timedelta(days=1)).weekday() in [5,6] else 0,
    feeder_code
]])

predicted_load = model.predict(input_features)[0]
deficit = predicted_load - available_supply

# ---------------- Dashboard ----------------
st.title("🏛 AI Energy Audit Command Center")

col1, col2, col3 = st.columns(3)
col1.metric("🔮 Predicted Load (MWh)", round(predicted_load,2))
col2.metric("⚡ Available Supply (MWh)", available_supply)
col3.metric("🚨 Power Deficit (MWh)", round(deficit,2))

# ---------------- Actual vs Predicted ----------------
st.subheader("📊 Actual vs Predicted Load")

comparison_df = pd.DataFrame({
    "Type": ["Actual (Yesterday)", "Predicted (Tomorrow)"],
    "Load": [last_row["Input_Energy_MWh"], predicted_load]
})

st.bar_chart(comparison_df.set_index("Type"))

# ---------------- Historical Trend ----------------
st.subheader("📈 Historical Load Trend")
st.line_chart(feeder_data.set_index("Date")["Input_Energy_MWh"])

# ---------------- Loss Analysis ----------------
st.subheader("📊 Feeder Loss Analysis")

data["Loss_%"] = ((data["Input_Energy_MWh"] - data["Billed_Energy_MWh"]) / data["Input_Energy_MWh"]) * 100

loss_rank = data.groupby("Feeder_ID")["Loss_%"].mean().sort_values(ascending=False).reset_index()
st.dataframe(loss_rank)

# ---------------- Loss Trend ----------------
st.subheader("📉 Loss Percentage Trend")

loss_trend = data.groupby("Date")["Loss_%"].mean().reset_index()
st.line_chart(loss_trend.set_index("Date"))

# ---------------- Feeder Comparison ----------------
st.subheader("⚡ Feeder Load Comparison")

latest_load = data.groupby("Feeder_ID")["Input_Energy_MWh"].last().reset_index()
st.bar_chart(latest_load.set_index("Feeder_ID"))

# ---------------- Supply vs Demand ----------------
st.subheader("🚨 Supply vs Demand")

supply_df = pd.DataFrame({
    "Category": ["Available Supply", "Predicted Demand"],
    "Value": [available_supply, predicted_load]
})

st.bar_chart(supply_df.set_index("Category"))

# ---------------- Anomaly Detection ----------------
threshold = data["Loss_%"].mean() + 2 * data["Loss_%"].std()
data["Anomaly"] = data["Loss_%"] > threshold

st.subheader("🚨 Theft / High Loss Alerts")
anomalies = data[data["Anomaly"] == True][["Date","Feeder_ID","Loss_%"]]
st.dataframe(anomalies.tail(10))

st.success("System Running Successfully ✅")
