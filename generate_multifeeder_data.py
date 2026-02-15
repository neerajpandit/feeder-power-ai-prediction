import pandas as pd
import numpy as np

def generate_multifeeder_data(
        start_date="2024-01-01",
        days=365,
        num_feeders=5,
        seed=42,
        save_path="multi_feeder_data.csv"
    ):

    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=days)
    all_data = []

    # 🌡️ Common seasonal temperature (same region)
    day_of_year = dates.dayofyear
    seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365)
    noise_temp = np.random.normal(0, 2, days)
    temperature_series = 25 + seasonal_temp + noise_temp

    for feeder_id in range(1, num_feeders + 1):

        base_load = np.random.randint(1000, 3000)
        temp_sensitivity_hot = np.random.randint(30, 60)
        temp_sensitivity_cold = np.random.randint(20, 50)

        feeder_data = pd.DataFrame()
        feeder_data["Date"] = dates
        feeder_data["Feeder_ID"] = f"F{feeder_id}"
        feeder_data["Temperature"] = temperature_series

        feeder_data["Day_of_week"] = feeder_data["Date"].dt.dayofweek
        feeder_data["Is_weekend"] = feeder_data["Day_of_week"].isin([5,6]).astype(int)

        # 🔥 Hot weather effect
        temp_effect = np.where(
            feeder_data["Temperature"] > 30,
            (feeder_data["Temperature"] - 30) * temp_sensitivity_hot,
            0
        )

        # ❄ Cold weather effect
        cold_effect = np.where(
            feeder_data["Temperature"] < 15,
            (15 - feeder_data["Temperature"]) * temp_sensitivity_cold,
            0
        )

        weekend_effect = feeder_data["Is_weekend"] * (-150)

        noise = np.random.normal(0, 80, days)

        feeder_data["Load"] = (
            base_load +
            temp_effect +
            cold_effect +
            weekend_effect +
            noise
        )

        feeder_data["Load"] = feeder_data["Load"].round(0)

        all_data.append(feeder_data)

    final_data = pd.concat(all_data)
    final_data = final_data[["Date", "Feeder_ID", "Temperature", "Load"]]

    final_data.to_csv(save_path, index=False)

    print(f"✅ Multi-feeder data saved as {save_path}")
    print(final_data.head())

    return final_data


if __name__ == "__main__":
    generate_multifeeder_data(days=730, num_feeders=10)
