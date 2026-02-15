import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib

# 1️⃣ Load Data
data = pd.read_csv("multi_feeder_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(["Feeder_ID", "Date"])

# 2️⃣ Feature Engineering (per feeder)
data["Load_yesterday"] = data.groupby("Feeder_ID")["Load"].shift(1)
data["Load_7day_avg"] = data.groupby("Feeder_ID")["Load"].rolling(7).mean().reset_index(0,drop=True)

data["Day_of_week"] = data["Date"].dt.dayofweek
data["Month"] = data["Date"].dt.month
data["Is_weekend"] = data["Day_of_week"].isin([5,6]).astype(int)

data = data.dropna()

# 3️⃣ Encode Feeder_ID
le = LabelEncoder()
data["Feeder_Code"] = le.fit_transform(data["Feeder_ID"])

# 4️⃣ Define Features
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
y = data["Load"]

# 5️⃣ Train-Test Split (time-based)
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 6️⃣ Train Advanced XGBoost Model
model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# 7️⃣ Prediction
pred = model.predict(X_test)

# 8️⃣ Evaluation
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print("MAE:", round(mae,2))
print("RMSE:", round(rmse,2))
print("MAPE:", round(mape,2), "%")

# 9️⃣ Save model & encoder
joblib.dump(model, "multi_feeder_model.pkl")
joblib.dump(le, "feeder_encoder.pkl")

print("✅ Model Saved Successfully")
