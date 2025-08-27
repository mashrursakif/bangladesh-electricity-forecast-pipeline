import requests
import pandas as pd
import numpy as np

# Get Weather Data

# Geo coords for Dhaka, Chittagong, and Patuakhali
dhk_coords = (23.7104, 90.40744)
cht_coords = (22.3384, 91.83168)
pat_coords = (22.36833, 90.3458)


def get_city_weather(coords):
    weather_api_url = f"https://api.open-meteo.com/v1/forecast?latitude={dhk_coords[0]}&longitude={dhk_coords[1]}&daily=relative_humidity_2m_mean,temperature_2m_max,temperature_2m_min,temperature_2m_mean,rain_sum,sunshine_duration&forecast_days=7"
    response = requests.get(weather_api_url)

    try:
        response.raise_for_status()
        data = response.json()

        return data
    except requests.exceptions.RequestException as e:
        print(e)


# Convert to pd DateFrame
def get_city_weather_df(daily_data):
    city_df = pd.DataFrame(
        columns=[
            "Date",  # Remove Date before passed to model
            "Rainfall",
            "Sunshine",
            "Humidity",
            "Temp_mean",
            "Temp_max",
            "Temp_min",
            "Year",
            "Month",
            "loadshed_prev",
            "generation_prev",
        ]
    )

    city_df["Date"] = daily_data["time"]
    city_df["Date"] = pd.to_datetime(city_df["Date"], format="%Y-%m-%d")
    city_df["Year"] = city_df["Date"].dt.year
    city_df["Month"] = city_df["Date"].dt.month

    city_df["Rainfall"] = daily_data["rain_sum"]
    city_df["Sunshine"] = daily_data["sunshine_duration"]
    # Convert seconds to hours
    city_df["Sunshine"] = city_df["Sunshine"] / (60 * 60)
    city_df["Humidity"] = daily_data["relative_humidity_2m_mean"]

    city_df["Temp_mean"] = daily_data["temperature_2m_mean"]
    city_df["Temp_max"] = daily_data["temperature_2m_max"]
    city_df["Temp_min"] = daily_data["temperature_2m_min"]

    return city_df


# Get weather data of all locations
dhk_data = get_city_weather(dhk_coords)
dhk_weather_df = get_city_weather_df(dhk_data["daily"])

cht_data = get_city_weather(cht_coords)
cht_weather_df = get_city_weather_df(cht_data["daily"])

pat_data = get_city_weather(pat_coords)
pat_weather_df = get_city_weather_df(pat_data["daily"])

all_weather_df = pd.concat([dhk_weather_df, cht_weather_df, pat_weather_df])

all_weather_df = (
    all_weather_df.groupby("Date")
    .agg(
        {
            "Rainfall": "mean",
            "Sunshine": "mean",
            "Humidity": "mean",
            "Temp_mean": "mean",
            "Temp_max": "max",
            "Temp_min": "min",
            "Year": "first",
            "Month": "first",
            "loadshed_prev": "first",
            "generation_prev": "first",
        }
    )
    .reset_index()
)

# Get Previous Day's Electricity Data
from datetime import datetime, timedelta

yesterday_date = datetime.today() - timedelta(days=1)

yesterday_date = yesterday_date.strftime("%d-%m-%Y")
yesterday_date

import io

pgcb_url = f"https://erp.pgcb.gov.bd/w/generations/view_generations?page={1}"

try:
    pgcb_res = requests.get(pgcb_url, verify=False)
    pgcb_res.raise_for_status()
    html = pgcb_res.text

    tables = pd.read_html(io.StringIO(html))

    yesterday_power_df = tables[0]

    yesterday_power_df.columns = yesterday_power_df.columns.droplevel(0)
    yesterday_power_df = yesterday_power_df[
        ["Date", "Generation(MW)", "Demand(MW)", "Loadshed"]
    ].copy()

    yesterday_power_df = yesterday_power_df[
        yesterday_power_df["Date"] == yesterday_date
    ]

    yesterday_power_df.rename(
        columns={
            "Generation(MW)": "Generation",
            "Demand(MW)": "Demand",
        },
        inplace=True,
    )

    # Convert units from MW to GW
    yesterday_power_df[["Generation", "Demand", "Loadshed"]] = (
        yesterday_power_df[["Generation", "Demand", "Loadshed"]] / 1000
    )
except requests.exceptions.RequestException as e:
    print(e)


# Create input data for model
input_df = all_weather_df.copy()


# Load Model
import lightgbm as lgb

generation_model = lgb.Booster(model_file="generation_lgbm_model.txt")
loadshed_model = lgb.Booster(model_file="loadshed_lgbm_model.txt")

# Set initial values
generation_previous = yesterday_power_df["Generation"].sum()
loadshed_previous = yesterday_power_df["Loadshed"].sum()

# Get Forecasts
forecasts = []

for idx in range(len(input_df)):
    row = input_df.iloc[idx].to_frame().T.infer_objects()

    row["generation_prev"] = generation_previous
    row["loadshed_prev"] = loadshed_previous

    generation_pred = generation_model.predict(row.drop(columns="Date"))[0]
    generation_previous = generation_pred

    loadshed_pred = loadshed_model.predict(row.drop(columns="Date"))[0]
    loadshed_previous = loadshed_pred

    daily_forecast = {
        "date": row["Date"].astype(str).values[0],
        "prediction": {
            "generation": generation_pred,
            "loadshed": loadshed_pred,
        },
    }
    forecasts.append(daily_forecast)

print(f"7 Days Forecast: {forecasts}")


# Add to predictions.json

from pathlib import Path
import json

pred_file = Path("predictions.json")

if pred_file.exists():
    with open(pred_file, "r") as f:
        preds = json.load(f)
else:
    preds = {"forecast": [], "history": []}

history = preds["history"]
previous_forecast = preds["forecast"]

yesterday = previous_forecast[0]
yesterday["label"] = {
    "generation": yesterday_power_df["Generation"].sum(),
    "loadshed": yesterday_power_df["Loadshed"].sum(),
}
history.append(yesterday)

# Update predictions
preds["history"] = history
preds["forecast"] = forecasts

try:
    with open(pred_file, "w") as f:
        json.dump(preds, f, indent=2)

    print("Successfully saved predictions")
except Exception as e:
    print(f"Error occurred: {e}")
