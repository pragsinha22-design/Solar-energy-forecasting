# Short-Term Solar Energy Generation Forecasting

## 1. Project Overview

This project focuses on forecasting solar photovoltaic (PV) AC power generation one hour ahead using historical power generation, weather conditions, irradiance, and time-based features.

The project uses 15-minute interval solar energy data and compares multiple forecasting approaches:

- Persistence Baseline
- Linear Regression
- ARIMA
- XGBoost

The final system also includes an interactive Streamlit dashboard for exploring the dataset, solar generation patterns, weather relationships, forecasts, and model performance.

---

## 2. Problem Statement

Solar power generation is highly dependent on weather conditions and time of day. Sudden changes in irradiance and weather can make short-term solar generation difficult to predict.

The objective of this project is to develop a machine learning and time-series forecasting system capable of predicting solar PV AC power generation one hour ahead using information available at the current time.

---

## 3. Objectives

The main objectives are:

1. Clean and preprocess solar power and weather datasets.
2. Align solar meter data with weather and irradiance data.
3. Perform exploratory data analysis (EDA).
4. Engineer useful time-series and weather-based features.
5. Create a one-hour-ahead forecasting target.
6. Compare baseline, statistical, linear and machine learning models.
7. Evaluate models using MAE, RMSE and R².
8. Perform data leakage checks.
9. Identify the best-performing forecasting model.
10. Develop an interactive Streamlit dashboard.

---

## 4. Dataset

The dataset contains approximately 23 months of 15-minute solar energy and weather observations.

### Data sources

The project uses three main datasets:

- Meter data
- Environment/weather data
- POA irradiance data

### Main variables

| Variable | Description |
|---|---|
| AC Power | Solar PV AC power output |
| Ambient Temperature | Recorded ambient temperature |
| Wind Speed | Recorded wind speed |
| Wind Direction | Wind direction |
| POA Irradiance | Plane-of-array solar irradiance |

The merged dataset contains approximately 67,183 records.

The data covers:

**February 2024 to December 2025**

---

## 5. Data Preprocessing

Several preprocessing steps were performed.

### Timestamp alignment

The meter timestamps were originally recorded in Pacific local time, while the weather datasets used UTC.

The meter timestamps were therefore converted from Pacific time to UTC while correctly handling daylight saving time.

### Irradiance resampling

The irradiance dataset contained mixed time intervals. It was resampled to a 15-minute resolution to match the meter data.

### Missing values

Short weather gaps were interpolated only when they were one hour or less.

Longer gaps were retained as missing rather than artificially generating weather information.

### Data merging

The datasets were merged using timestamps after time-zone alignment.

The meter dataset was treated as the main dataset because AC power is the forecasting target.

---

## 6. Exploratory Data Analysis

EDA was performed to understand the relationships between solar generation and environmental variables.

Important observations include:

- Irradiance has the strongest relationship with AC power.
- Correlation between AC power and POA irradiance is approximately **0.974**.
- Average power reaches its highest level around midday.
- Approximately 51% of observations have zero AC power, corresponding mainly to nighttime.
- Wind speed has very little linear correlation with power.
- Temperature has a moderate overall correlation with power, partly because higher temperatures occur during daylight hours.
- Power output shows a maximum near 705–708, suggesting possible system output limitation/clipping.

EDA visualizations include:

- Power over time
- Power vs irradiance
- Power vs temperature
- Power vs wind speed
- Average power by hour
- Correlation heatmap
- Power distribution
- Representative daily profiles

---

## 7. Forecasting Target

The forecasting problem is defined as:

> Given information available at time t, predict AC power at time t + 1 hour.

Since the data is at 15-minute resolution, one hour corresponds to four 15-minute steps.

The target variable is:

`power_1h_ahead`

The target was generated using timestamp-based alignment to ensure that the correct future value was used.

---

## 8. Feature Engineering

The final model uses 14 input features.

### Historical power features

- `ac_power`
- `power_lag_1`
- `power_lag_2`
- `power_lag_4`

These represent current power and historical power at 15-minute, 30-minute and 60-minute intervals.

### Weather features

- `ambient_temperature`
- `wind_speed`
- `wind_direction`
- `poa_irradiance`

### Time features

- `hour`
- `day`
- `month`
- `day_of_week`

### Target-time cyclic features

- `target_hour_sin`
- `target_hour_cos`

The cyclic features represent the time of day being forecast and allow models to represent the circular relationship between hours such as 23:00 and 00:00.

---

## 9. Data Leakage Prevention

Data leakage was specifically checked because this is a time-series forecasting problem.

The project includes automated leakage checks.

The checks verify:

- Correct lag calculations
- Correct target calculations
- No future information used in features
- Correct chronological train/validation/test split
- No overlap between datasets
- No interpolated weather rows included in the final modelling dataset
- Correct handling of the one-hour-ahead target

All automated leakage checks passed.

---

## 10. Train / Validation / Test Split

The data was divided chronologically rather than randomly.

The split was approximately:

- 70% Training
- 15% Validation
- 15% Testing

This prevents future observations from being used to train the model.

The test period occurs later in time and therefore also represents a seasonal shift toward autumn/winter conditions.

---

## 11. Models Used

Four approaches were evaluated.

### 11.1 Persistence Baseline

The persistence model assumes that recent power remains similar in the near future.

It provides a simple baseline against which more advanced models can be compared.

### 11.2 Linear Regression

Linear Regression models the relationship between the engineered features and future AC power.

### 11.3 ARIMA

ARIMA is a statistical time-series forecasting approach.

It uses historical AC power and does not directly use weather variables.

### 11.4 XGBoost

XGBoost is a gradient-boosted decision-tree machine learning model.

It can model nonlinear relationships between historical power, irradiance, weather and time-based features.

---

## 12. Evaluation Metrics

The models were evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted power.

Lower MAE is better.

### Root Mean Squared Error (RMSE)

Penalizes larger prediction errors more strongly than MAE.

Lower RMSE is better.

### R² Score

Measures how much of the variation in the target is explained by the model.

Higher R² is better.

---

## 13. Model Results

The final test-set results are:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **XGBoost** | **26.54** | **58.35** | **0.8905** |
| Linear Regression | 37.29 | 66.23 | 0.8589 |
| ARIMA | 47.52 | 71.75 | 0.8344 |
| Persistence Baseline | 39.74 | 80.98 | 0.7891 |

### Best Model

**XGBoost achieved the best performance.**

It obtained:

- MAE = **26.54**
- RMSE = **58.35**
- R² = **0.8905**

Therefore, XGBoost was selected as the best-performing model for the one-hour-ahead solar power forecasting task.

---

## 14. XGBoost Feature Importance

The most important features identified by XGBoost include:

| Feature | Importance |
|---|---:|
| `ac_power` | 0.497 |
| `target_hour_cos` | 0.216 |
| `hour` | 0.110 |
| `target_hour_sin` | 0.091 |
| `poa_irradiance` | 0.032 |

Current AC power is the most important feature, while time-of-day information and irradiance also contribute significantly to the prediction.

---

## 15. Streamlit Dashboard

An interactive Streamlit dashboard was developed for the project.

The dashboard provides:

- Dataset summary
- Solar generation visualization
- Weather and irradiance visualization
- Average power by hour
- Correlation analysis
- Actual vs predicted power
- Forecast output
- Model performance comparison
- Model selection
- Representative forecasting periods

The dashboard can be launched using:

```bash
python -m streamlit run dashboard/app.py
