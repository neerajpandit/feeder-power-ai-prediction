import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import joblib

# Load dataset
data = pd.read_csv("energy_feeder_dataset.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Feeder_ID","Date"])

# Feature Engineering
data["Load_yesterday"] = data.groupby("Feeder_ID")["Input_Energy_MWh"].shift(1)
data["Load_7day_avg"] = data.groupby("Feeder_ID")["Input_Energy_MWh"].rolling(7).mean().reset_index(0,drop=True)

data["Day_of_week"] = data["Date"].dt.dayofweek
data["Month"] = data["Date"].dt.month
data["Is_weekend"] = data["Day_of_week"].isin([5,6]).astype(int)

data = data.dropna()

# Encode feeder
le = LabelEncoder()
data["Feeder_Code"] = le.fit_transform(data["Feeder_ID"])

features = [
    "Temperature",
    "Load_yesterday",
    "Load_7day_avg",
    "Day_of_week",
    "Month",
    "Is_weekend",
    "Feeder_Code"
]

X = data[features]
y = data["Input_Energy_MWh"]

split = int(len(data)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=-1
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
print("MAE:", round(mae,2))

joblib.dump(model, "energy_model.pkl")
joblib.dump(le, "feeder_encoder.pkl")

print("✅ Energy Forecast Model Saved")
