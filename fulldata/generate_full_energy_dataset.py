import pandas as pd
import numpy as np

def generate_energy_dataset(
        start_date="2024-01-01",
        days=365,
        num_feeders=5,
        seed=42,
        save_path="energy_feeder_dataset.csv"
    ):

    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=days)

    all_data = []

    for feeder in range(1, num_feeders + 1):

        feeder_id = f"F{feeder}"
        base_load = np.random.randint(800, 2000)
        base_loss_percent = np.random.uniform(15, 25)

        for date in dates:

            day_of_year = date.dayofyear

            # Seasonal Temperature
            temperature = 5 + 15 * np.sin(2 * np.pi * day_of_year / 365)
            temperature += np.random.normal(0, 2)

            # Temperature Effect on Load
            if temperature < 0:
                temp_effect = abs(temperature) * 40
            elif temperature > 30:
                temp_effect = (temperature - 30) * 30
            else:
                temp_effect = 0

            input_energy = base_load + temp_effect + np.random.normal(0, 50)

            # Normal Loss
            loss_percent = base_loss_percent + np.random.normal(0, 2)

            # Random Theft Spike (simulate anomaly)
            if np.random.rand() < 0.03:
                loss_percent += np.random.uniform(10, 20)

            billed_energy = input_energy * (1 - loss_percent / 100)

            all_data.append([
                date,
                feeder_id,
                round(temperature,2),
                round(input_energy,2),
                round(billed_energy,2),
                round(loss_percent,2)
            ])

    df = pd.DataFrame(all_data, columns=[
        "Date",
        "Feeder_ID",
        "Temperature",
        "Input_Energy_MWh",
        "Billed_Energy_MWh",
        "Loss_%"
    ])

    df.to_csv(save_path, index=False)
    print(f"✅ Dataset saved as {save_path}")
    print(df.head())

    return df


if __name__ == "__main__":
    generate_energy_dataset(days=730, num_feeders=8)
