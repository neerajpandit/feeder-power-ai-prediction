import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib

# 1️⃣ Load Data
data = pd.read_csv("data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')

# 2️⃣ Feature Engineering
data['Load_yesterday'] = data['Load'].shift(1)
data['Load_7day_avg'] = data['Load'].rolling(7).mean()
data['Day_of_week'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['Is_weekend'] = data['Day_of_week'].isin([5,6]).astype(int)

data = data.dropna()

# 3️⃣ Define Features
features = [
    'Temperature',
    'Load_yesterday',
    'Load_7day_avg',
    'Day_of_week',
    'Month',
    'Is_weekend'
]

X = data[features]
y = data['Load']

# 4️⃣ Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 5️⃣ Train Model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)

model.fit(X_train, y_train)

# 6️⃣ Prediction
predictions = model.predict(X_test)

# 7️⃣ Evaluation
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)

# 8️⃣ Plot Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Load")
plt.show()

# 9️⃣ Save Model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
