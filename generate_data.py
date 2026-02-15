import pandas as pd
import numpy as np

def generate_electricity_data(
        start_date="2024-01-01",
        days=770,
        base_load=1500,
        temp_base=25,
        seed=42,
        save_path="data.csv"
    ):
    """
    Generate synthetic electricity load dataset with temperature impact.
    
    Parameters:
    - start_date: starting date (YYYY-MM-DD)
    - days: number of days to generate
    - base_load: base electricity load
    - temp_base: average temperature
    - seed: random seed for reproducibility
    - save_path: CSV file name
    
    Output:
    - Saves CSV file
    - Returns DataFrame
    """

    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=days)

    data = pd.DataFrame()
    data["Date"] = dates

    # 🎯 Seasonal Temperature Pattern (sin wave)
    day_of_year = data["Date"].dt.dayofyear
    seasonal_temp = 10 * np.sin(2 * np.pi * day_of_year / 365)
    noise_temp = np.random.normal(0, 2, days)

    data["Temperature"] = temp_base + seasonal_temp + noise_temp

    # 🎯 Weekend Effect
    data["Day_of_week"] = data["Date"].dt.dayofweek
    data["Is_weekend"] = data["Day_of_week"].isin([5, 6]).astype(int)

    # 🎯 Load Calculation Logic
    # Load increases when temp > 30 (AC use)
    temp_effect = np.where(
        data["Temperature"] > 30,
        (data["Temperature"] - 30) * 40,
        0
    )

    # Winter heating effect
    cold_effect = np.where(
        data["Temperature"] < 15,
        (15 - data["Temperature"]) * 30,
        0
    )

    weekend_effect = data["Is_weekend"] * (-100)

    random_noise = np.random.normal(0, 50, days)

    data["Load"] = (
        base_load +
        temp_effect +
        cold_effect +
        weekend_effect +
        random_noise
    )

    data["Load"] = data["Load"].round(0)

    # Keep only required columns
    final_data = data[["Date", "Temperature", "Load"]]

    # Save CSV
    final_data.to_csv(save_path, index=False)

    print(f"✅ Data generated successfully! Saved as {save_path}")
    print(final_data.head())

    return final_data


# Run function if file executed directly
if __name__ == "__main__":
    generate_electricity_data(days=770)   # 2 years data
