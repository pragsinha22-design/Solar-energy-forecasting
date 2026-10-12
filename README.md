# ☀️ Short-Term Solar Energy Generation Forecasting

### A Machine Learning and Time-Series Forecasting System for Solar PV Power Generation

An end-to-end solar energy forecasting project that predicts **solar photovoltaic (PV) AC power generation one hour ahead** using historical power output, weather conditions, solar irradiance, and time-based features.

The project combines data preprocessing, exploratory data analysis (EDA), feature engineering, time-series analysis, machine learning, model evaluation, and an interactive **Streamlit dashboard** to analyze solar generation patterns and compare forecasting models.

The system evaluates four forecasting approaches—Persistence Baseline, Linear Regression, ARIMA, and XGBoost—and identifies XGBoost as the best-performing model on the held-out test dataset.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Problem Statement](#-problem-statement)
* [Objectives](#-objectives)
* [Key Features](#-key-features)
* [Technology Stack](#-technology-stack)
* [Dataset Description](#-dataset-description)
* [Project Workflow](#-project-workflow)
* [Data Preprocessing](#-data-preprocessing)
* [Exploratory Data Analysis](#-exploratory-data-analysis)
* [Feature Engineering](#-feature-engineering)
* [Forecasting Methodology](#-forecasting-methodology)
* [Models Implemented](#-models-implemented)
* [Model Evaluation](#-model-evaluation)
* [Results and Analysis](#-results-and-analysis)
* [Feature Importance](#-feature-importance)
* [Streamlit Dashboard](#-streamlit-dashboard)
* [Installation and Setup](#-installation-and-setup)
* [Project Structure](#-project-structure)
* [Data Leakage Prevention](#-data-leakage-prevention)
* [Applications](#-applications)
* [Limitations](#-limitations)
* [Future Enhancements](#-future-enhancements)
* [Conclusion](#-conclusion)

---

## 📖 Project Overview

Solar energy generation depends on several environmental factors, including solar irradiance, weather conditions, temperature, and the time of day. These factors can change over time, making accurate short-term forecasting important for understanding and managing solar power production.

This project develops a forecasting pipeline that uses historical solar meter readings, weather observations, and irradiance measurements to predict future AC power generation.

The system processes approximately **67,183 observations spanning February 2024 to December 2025**, recorded at 15-minute intervals.

The forecasting task is defined as predicting AC power at time `t + 1 hour` using information available at time `t`.

Four forecasting approaches are compared to understand the performance of simple baselines, statistical time-series methods, linear models, and gradient-boosted decision trees.

An interactive Streamlit dashboard makes it possible to explore the dataset, visualize generation patterns, examine environmental relationships, and compare model predictions with actual power output.

### Project Highlights

* ☀️ One-hour-ahead solar power forecasting
* 📊 Approximately 67,183 merged observations
* 🌦️ Integration of solar meter, weather, and irradiance datasets
* 🧹 Timestamp alignment and missing-value handling
* ⚙️ Time-series feature engineering
* 🤖 Comparison of four forecasting approaches
* 📈 Chronological training, validation, and test splits
* 🔍 Automated data leakage checks
* 🏆 Best test-set R² of **0.8905** using XGBoost
* 🖥️ Interactive Streamlit dashboard

---

## 🎯 Problem Statement

Solar photovoltaic power output varies with irradiance, environmental conditions, and time of day. These variations make it challenging to estimate future generation accurately.

The objective of this project is to build a reliable short-term forecasting system that predicts solar PV AC power generation one hour ahead using historical generation measurements and relevant weather and time-based information.

The project also evaluates different forecasting approaches to identify which model performs best on unseen chronological test data.

---

## 🎯 Objectives

1. Collect and integrate solar meter, weather, and irradiance datasets.
2. Clean and preprocess the collected data.
3. Align timestamps across datasets recorded in different time zones.
4. Perform exploratory data analysis to understand solar generation patterns.
5. Engineer meaningful historical, weather, and time-based features.
6. Construct a one-hour-ahead forecasting target.
7. Implement Persistence, Linear Regression, ARIMA, and XGBoost forecasting approaches.
8. Evaluate model performance using MAE, RMSE, and R².
9. Prevent data leakage through chronological splitting and automated checks.
10. Identify the best-performing forecasting model.
11. Develop an interactive Streamlit dashboard for data exploration and forecasting analysis.

---

## ✨ Key Features

### 1. Data Integration and Preprocessing

* Integration of solar meter, environmental, and POA irradiance data.
* Time-zone conversion and timestamp alignment.
* Resampling irradiance observations to a 15-minute interval.
* Controlled interpolation of short weather-data gaps.
* Missing-value handling and preparation of the final modeling dataset.

### 2. Exploratory Data Analysis

* Solar power generation over time.
* Relationship between AC power and solar irradiance.
* Analysis of temperature and wind conditions.
* Hourly solar generation patterns.
* Power distribution and correlation analysis.
* Representative daily generation profiles.

### 3. Time-Series Feature Engineering

* Historical AC power measurements.
* Lagged power features.
* Weather and irradiance variables.
* Calendar and time-based features.
* Cyclic encoding of the forecast time of day.

### 4. Multiple Forecasting Models

Four approaches are implemented and evaluated using the same forecasting objective:

* Persistence Baseline
* Linear Regression
* ARIMA
* XGBoost

### 5. Model Evaluation and Comparison

Models are compared using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination (R²)

### 6. Interactive Dashboard

The Streamlit dashboard allows users to explore solar generation, visualize weather relationships, inspect forecasting results, and compare model performance.

---

## 🛠️ Technology Stack

| Technology      | Purpose                                    |
| --------------- | ------------------------------------------ |
| Python          | Core programming language                  |
| Pandas          | Data manipulation and preprocessing        |
| NumPy           | Numerical computations                     |
| Matplotlib      | Data visualization                         |
| Seaborn         | Statistical visualization                  |
| Scikit-learn    | Linear Regression and model evaluation     |
| XGBoost         | Gradient-boosted tree forecasting          |
| Statsmodels     | ARIMA time-series modeling                 |
| Streamlit       | Interactive web dashboard                  |
| Plotly, if used | Interactive charts and visualizations      |
| Git             | Version control                            |
| GitHub          | Source code management and project hosting |

*The technology list should reflect the libraries actually imported and used in your project.*

---

## 📂 Dataset Description

The project uses three primary datasets.

| Dataset             | Description                                                                    |
| ------------------- | ------------------------------------------------------------------------------ |
| Meter Data          | Contains solar PV AC power measurements and timestamps                         |
| Environment Data    | Contains environmental observations, including temperature and wind conditions |
| POA Irradiance Data | Contains plane-of-array solar irradiance measurements                          |

### Dataset Summary

| Property                 | Details                                           |
| ------------------------ | ------------------------------------------------- |
| Application              | Solar PV power forecasting                        |
| Data frequency           | 15 minutes                                        |
| Approximate observations | 67,183                                            |
| Data coverage            | February 2024 – December 2025                     |
| Target variable          | `power_1h_ahead`                                  |
| Forecast horizon         | 1 hour                                            |
| Forecasting type         | Supervised regression and time-series forecasting |

### Important Variables

| Variable              | Description                                     |
| --------------------- | ----------------------------------------------- |
| `ac_power`            | Current solar PV AC power output                |
| `ambient_temperature` | Recorded ambient temperature                    |
| `wind_speed`          | Recorded wind speed                             |
| `wind_direction`      | Recorded wind direction                         |
| `poa_irradiance`      | Plane-of-array solar irradiance                 |
| `power_lag_1`         | AC power one 15-minute interval earlier         |
| `power_lag_2`         | AC power two 15-minute intervals earlier        |
| `power_lag_4`         | AC power four 15-minute intervals earlier       |
| `hour`                | Hour of the current observation                 |
| `day`                 | Day of the month                                |
| `month`               | Month of the year                               |
| `day_of_week`         | Day of the week                                 |
| `target_hour_sin`     | Sine encoding of the forecast hour              |
| `target_hour_cos`     | Cosine encoding of the forecast hour            |
| `power_1h_ahead`      | AC power one hour after the current observation |

**Note:** The dataset contains approximately 67,183 merged observations. The number of records available for modeling can be smaller after removing rows with unavailable targets, missing required features, and observations affected by lag or forecasting alignment.

---

## 🔄 Project Workflow

The project follows a structured machine learning workflow.

1. **Data Collection:** Load the solar meter, weather, and irradiance datasets.
2. **Data Preprocessing:** Clean the data, standardize timestamps, and handle missing values.
3. **Data Integration:** Merge the datasets using aligned timestamps.
4. **Exploratory Data Analysis:** Examine solar generation trends and environmental relationships.
5. **Feature Engineering:** Generate historical power, weather, calendar, and cyclic features.
6. **Target Construction:** Align the target with AC power measured one hour into the future.
7. **Data Splitting:** Divide observations chronologically into training, validation, and test sets.
8. **Model Training:** Train and evaluate the selected forecasting approaches.
9. **Performance Evaluation:** Compare predictions using MAE, RMSE, and R².
10. **Leakage Verification:** Check feature timing, target alignment, and dataset boundaries.
11. **Dashboard Development:** Present visualizations and forecasting results through Streamlit.

---

## 🧹 Data Preprocessing

Data preprocessing is an important part of the project because the three datasets may differ in timestamp conventions, observation frequency, and data completeness.

### 1. Timestamp Alignment

The meter timestamps were originally recorded in Pacific local time, whereas the weather datasets used UTC.

The meter timestamps were converted from Pacific local time to UTC using time-zone-aware conversion to account for daylight saving time.

This step ensures that observations from different sources are aligned to the correct time.

### 2. Irradiance Resampling

The POA irradiance dataset contained observations at mixed time intervals.

The irradiance measurements were resampled to a 15-minute resolution to align with the solar meter data.

### 3. Missing-Value Handling

Short weather-data gaps of up to one hour were interpolated.

Longer gaps were retained as missing rather than filling them with potentially unreliable estimated weather conditions.

### 4. Dataset Merging

The meter dataset was treated as the primary dataset because AC power is the target variable.

Weather and irradiance observations were merged with the meter data using the aligned timestamps.

### 5. Final Modeling Dataset

The processed dataset was prepared for exploratory analysis, feature engineering, and model training.

The final modeling data excludes observations that do not satisfy the required feature and target availability conditions.

---

## 📊 Exploratory Data Analysis

Exploratory Data Analysis was conducted to understand the behavior of solar power generation and its relationship with environmental conditions.

### Key Findings

**1. Solar Irradiance and AC Power**

POA irradiance has a strong positive relationship with AC power.

The observed correlation is approximately **0.974**, indicating that irradiance is strongly associated with solar generation in the analyzed dataset.

**2. Daily Generation Patterns**

Average solar power reaches its highest levels around midday, when solar irradiance is generally greater.

**3. Nighttime Generation**

Approximately 51% of the observations have zero AC power. This is primarily associated with nighttime periods when solar generation is unavailable.

**4. Temperature Relationship**

Ambient temperature has a moderate overall relationship with power output. Part of this relationship may arise because both temperature and solar generation tend to increase during daylight hours.

**5. Wind Speed**

Wind speed exhibits very little linear correlation with AC power in the analyzed dataset.

**6. Possible Output Limitation**

The maximum observed power is approximately 705–708 in the dataset's reported power units. This pattern may indicate an output limit or clipping, but confirming that interpretation requires additional information about the PV system's rated capacity and inverter configuration.

### Visualizations

The analysis includes:

* Power generation over time.
* AC power versus POA irradiance.
* AC power versus ambient temperature.
* AC power versus wind speed.
* Average power generation by hour.
* Feature correlation heatmap.
* AC power distribution.
* Representative daily generation profiles.

These visualizations help identify generation patterns, relationships between variables, and potential limitations in the underlying data.

---

## ⚙️ Feature Engineering

Feature engineering converts the raw time-series observations into useful inputs for forecasting.

The final model uses **14 input features**.

### 1. Historical Power Features

| Feature       | Description              |
| ------------- | ------------------------ |
| `ac_power`    | Current AC power         |
| `power_lag_1` | Power 15 minutes earlier |
| `power_lag_2` | Power 30 minutes earlier |
| `power_lag_4` | Power 60 minutes earlier |

These features provide information about the current generation level and its recent history.

### 2. Weather Features

| Feature               | Description                     |
| --------------------- | ------------------------------- |
| `ambient_temperature` | Ambient temperature             |
| `wind_speed`          | Wind speed                      |
| `wind_direction`      | Wind direction                  |
| `poa_irradiance`      | Plane-of-array solar irradiance |

These features represent environmental conditions that may influence solar generation.

### 3. Time-Based Features

| Feature       | Description                     |
| ------------- | ------------------------------- |
| `hour`        | Hour of the current observation |
| `day`         | Day of the month                |
| `month`       | Month of the year               |
| `day_of_week` | Day of the week                 |

These variables help the models capture daily and seasonal patterns.

### 4. Cyclic Forecast-Time Features

Two additional features encode the hour being forecast:

* `target_hour_sin`
* `target_hour_cos`

Cyclic encoding represents the circular nature of time. For example, 23:00 and 00:00 are close in time, even though their numerical hour values differ substantially.

The cyclic representation helps models capture this relationship.

---

## 🔮 Forecasting Methodology

The project forecasts solar AC power one hour ahead.

Because the original observations are recorded at 15-minute intervals, the forecast horizon corresponds to four sampling intervals.

The target variable is:

`power_1h_ahead`

Conceptually, the task is:

**Input:** Information available at time `t`.

**Output:** Predicted AC power at time `t + 1 hour`.

The forecasting pipeline uses historical power, weather conditions, irradiance, and time-based features to estimate future generation.

### Chronological Data Split

The dataset was divided into three chronological subsets.

| Dataset    | Approximate Share | Purpose                                    |
| ---------- | ----------------: | ------------------------------------------ |
| Training   |               70% | Fit the models                             |
| Validation |               15% | Compare models and support model selection |
| Testing    |               15% | Evaluate final generalization performance  |

A chronological split preserves the direction of time and avoids the problems associated with randomly mixing past and future observations.

The held-out test period occurs later in time than the training data and includes a shift toward autumn and winter conditions.

This provides a more realistic assessment of how the forecasting approaches perform on later observations.

---

## 🤖 Models Implemented

Four forecasting approaches were evaluated.

### 1. Persistence Baseline

The Persistence Baseline assumes that a recent observed power value remains a useful estimate of near-future power.

It provides a simple reference point for determining whether more complex forecasting models improve prediction accuracy.

### 2. Linear Regression

Linear Regression estimates a linear relationship between the input features and the future AC power target.

It provides an interpretable baseline and helps evaluate how much predictive information can be captured through linear relationships.

### 3. ARIMA

ARIMA stands for AutoRegressive Integrated Moving Average.

It is a statistical time-series forecasting method that models patterns in historical observations.

In this project, ARIMA uses historical AC power rather than directly incorporating the weather variables used by the feature-based models.

### 4. XGBoost

XGBoost is a gradient-boosted decision-tree algorithm that builds an ensemble of trees to model relationships between input features and a target variable.

It can capture nonlinear patterns and interactions between historical power, weather conditions, irradiance, and time-based features.

Among the four approaches evaluated, XGBoost achieved the best test-set results.

---

## 📈 Model Evaluation

The forecasting models were evaluated using three standard regression metrics.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted power.

A lower MAE indicates that predictions are closer to the observed values on average.

### Root Mean Squared Error (RMSE)

RMSE measures prediction error while assigning greater weight to larger errors.

A lower RMSE indicates better performance under this metric.

### Coefficient of Determination (R²)

R² measures predictive performance relative to a baseline that predicts the mean of the observed target values.

A higher R² generally indicates that the predictions explain more of the variation in the target.

The units of MAE and RMSE are the same as those of the AC power target.

---

## 🏆 Results and Analysis

The following table presents the final test-set results.

| Rank | Model                |     MAE ↓ |    RMSE ↓ |       R² ↑ |
| ---: | -------------------- | --------: | --------: | ---------: |
|    1 | **XGBoost**          | **26.54** | **58.35** | **0.8905** |
|    2 | Linear Regression    |     37.29 |     66.23 |     0.8589 |
|    3 | ARIMA                |     47.52 |     71.75 |     0.8344 |
|    4 | Persistence Baseline |     39.74 |     80.98 |     0.7891 |

*Lower MAE and RMSE are better. Higher R² is better. All metrics are calculated on the reported test set.*

### Best-Performing Model: XGBoost

XGBoost achieved the best overall test-set performance among the four evaluated approaches.

Its results were:

* **MAE:** 26.54
* **RMSE:** 58.35
* **R²:** 0.8905

The model obtained the lowest MAE and RMSE and the highest R² in the comparison.

Its performance suggests that the combination of historical power measurements, environmental variables, and time-based features provides useful information for predicting one-hour-ahead solar generation.

### Comparison with Linear Regression

Compared with Linear Regression, XGBoost reduced MAE by approximately 28.8% and RMSE by approximately 11.9%.

This indicates that the tree-based model captured predictive relationships that the linear model did not represent as effectively.

### Comparison with the Persistence Baseline

XGBoost also achieved a lower MAE and RMSE than the Persistence Baseline, along with a higher R².

This supports the use of a more sophisticated forecasting model for the analyzed dataset.

### Interpretation

The results show that XGBoost is the strongest-performing model in this experiment.

However, the reported metrics apply to the particular dataset, feature set, evaluation period, and preprocessing pipeline used in this project. They should not be interpreted as a guarantee of equivalent performance on other solar installations or future operating conditions.

---

## 🔍 XGBoost Feature Importance

Feature importance analysis was used to identify which inputs contributed most to the trained XGBoost model's predictions.

| Rank | Feature           | Importance |
| ---: | ----------------- | ---------: |
|    1 | `ac_power`        |      0.497 |
|    2 | `target_hour_cos` |      0.216 |
|    3 | `hour`            |      0.110 |
|    4 | `target_hour_sin` |      0.091 |
|    5 | `poa_irradiance`  |      0.032 |

### Key Observations

* **Current AC power** is the most important feature, highlighting the predictive value of the current generation level.
* **Forecast-time features** contribute substantially, indicating that daily generation patterns provide useful predictive information.
* **POA irradiance** also contributes to the model's predictions.
* Other weather and historical features provide additional information to the forecasting pipeline.

Feature importance describes the model's learned predictive behavior. It does not establish that a feature independently causes changes in solar power generation.

The importance values are specific to the trained model and its feature-importance calculation method.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard built using Streamlit.

The dashboard presents the dataset, solar generation patterns, environmental relationships, forecasting results, and model comparisons through a unified interface.

### Dashboard Features

| Feature                  | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| Dataset Summary          | Displays an overview of the available dataset                 |
| Generation Visualization | Explores AC power trends over time                            |
| Weather Analysis         | Examines temperature and wind relationships                   |
| Irradiance Analysis      | Visualizes the relationship between irradiance and generation |
| Hourly Analysis          | Displays average generation by hour                           |
| Correlation Analysis     | Examines relationships between numerical variables            |
| Actual vs. Predicted     | Compares model predictions with observed power                |
| Forecast Output          | Presents forecasting results                                  |
| Model Comparison         | Compares model performance using evaluation metrics           |
| Model Selection          | Supports exploration of the available forecasting approaches  |
| Representative Periods   | Examines model behavior over selected periods                 |

### Run the Dashboard

From the project's root directory, execute:

```bash
python -m streamlit run dashboard/app.py
```

Streamlit will start the local application and display the address where the dashboard can be opened in a browser.

---

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

Make sure the following are installed:

* Python
* pip
* Git

### Step 1: Clone the Repository

Replace `<your-username>` and `<your-repository>` with your GitHub username and repository name.

```bash
git clone https://github.com/<your-username>/<your-repository>.git
```

Navigate into the project directory:

```bash
cd <your-repository>
```

### Step 2: Create a Virtual Environment

On Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

If the repository includes a `requirements.txt` file, install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

If a dependency file has not yet been created, generate one containing the packages required by the project's actual source code.

### Step 4: Prepare the Dataset

Place the required meter, environment, and POA irradiance datasets in the expected data directory.

Check the repository's data-loading scripts or configuration to confirm the required filenames and paths.

If the raw datasets are not included in the repository, obtain them separately and place them in the appropriate locations before running the preprocessing pipeline.

### Step 5: Run the Dashboard

From the root directory:

```bash
python -m streamlit run dashboard/app.py
```

Open the local address displayed in the terminal to interact with the dashboard.

**Note:** The dashboard and model pipeline must be configured to load the appropriate processed data and trained models. Additional setup may be required depending on how these artifacts are stored in the repository.

---

## 📁 Project Structure

The following is a suggested structure for organizing the repository. Adjust the filenames to match your actual implementation.

```text
Short-Term-Solar-Energy-Forecasting/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── models/
│   ├── xgboost_model.*
│   └── other_model_artifacts.*
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── leakage_checks.py
│
├── tests/
│   └── test_leakage.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

This layout is illustrative rather than a description of verified repository contents.

Keep large datasets, generated model artifacts, and temporary files out of Git unless they are necessary and appropriate to distribute. If the project uses different directories or filenames, document the actual structure instead.

---

## 🛡️ Data Leakage Prevention

Data leakage is a significant risk in time-series forecasting because future observations can inadvertently influence training or evaluation.

This project includes automated checks designed to verify the integrity of the forecasting pipeline.

### Leakage Checks

* **Lag Validation:** Verifies that historical power features are calculated from the appropriate earlier observations.
* **Target Alignment:** Checks that the target corresponds to AC power one hour after the forecast origin.
* **Feature Timing:** Verifies that future observations are not unintentionally used to construct prediction features.
* **Chronological Splitting:** Ensures that training, validation, and test observations are ordered by time.
* **Split Separation:** Checks that the datasets do not overlap in ways that invalidate the evaluation.
* **Weather Interpolation:** Excludes interpolated weather rows from the final modeling dataset, according to the project's preprocessing rules.
* **Forecast Horizon:** Verifies the alignment of the one-hour-ahead target.

All automated leakage checks passed in the reported experiment.

These checks support the reliability of the evaluation, although their effectiveness depends on the cases covered by the implemented tests.

---

## 🌍 Applications

Short-term solar power forecasting can support several renewable-energy applications.

### 1. Solar Power Plant Monitoring

Forecasts can help operators compare expected and observed generation and investigate unusual deviations.

### 2. Energy Management

Predicted generation can support planning for energy consumption, storage, and other resources.

### 3. Smart Grids

Short-term generation estimates can provide useful information for balancing electricity supply and demand.

### 4. Battery Storage Planning

Forecasts may help estimate periods of higher or lower solar availability when planning battery charging and discharging.

### 5. Renewable Energy Analysis

Historical data and forecasting results can help researchers investigate daily generation patterns and relationships between weather conditions and solar power output.

These are potential applications of short-term solar forecasting rather than claims of functionality already implemented in the dashboard.

---

## ⚠️ Limitations

Although the XGBoost model achieved strong results on the test dataset, several limitations should be considered.

* The model was evaluated using a particular solar generation dataset and may not generalize directly to other installations.
* The test period includes different seasonal conditions from the earlier training period.
* Forecast accuracy depends on the availability and quality of the historical power, weather, and irradiance measurements.
* The reported feature-importance values do not establish causal relationships.
* ARIMA does not directly incorporate the weather variables used by the feature-based models.
* A forecast one hour ahead may not capture every rapid change caused by passing clouds or sudden environmental changes.
* The project does not establish performance under all possible weather conditions or across multiple independent solar installations.
* The interpretation of the apparent maximum power level requires additional system-capacity information.

Further evaluation across different seasonal periods and operating conditions would help establish how consistently the model performs.

---

## 🔮 Future Enhancements

The project can be extended in several ways.

* **Advanced Forecasting Models:** Evaluate LightGBM, CatBoost, Random Forest, and other suitable regression models.
* **Deep Learning:** Explore LSTM, GRU, and Temporal Convolutional Networks for time-series forecasting.
* **Weather Forecast Integration:** Incorporate weather forecasts available at prediction time rather than relying exclusively on current observations.
* **Multi-Horizon Forecasting:** Extend the system to predict power 15 minutes, 30 minutes, one hour, and several hours ahead.
* **Forecast Uncertainty:** Add prediction intervals or quantile forecasts to represent uncertainty.
* **Explainable AI:** Use SHAP or related techniques to analyze feature contributions at both global and individual-prediction levels.
* **Model Retraining:** Develop a scheduled retraining pipeline to adapt to changing seasonal and operating conditions.
* **Deployment:** Deploy the dashboard to a suitable hosting platform.
* **Model Monitoring:** Track prediction errors and detect changes in data distributions over time.
* **Expanded Evaluation:** Validate the forecasting system on additional time periods and independent solar installations.

These enhancements could improve the usefulness, interpretability, and robustness of the forecasting system.

---

## ✅ Conclusion

The Short-Term Solar Energy Generation Forecasting project demonstrates an end-to-end approach to predicting solar PV AC power generation one hour ahead using historical power measurements, weather observations, irradiance, and time-based features.

The project combines data preprocessing, timestamp alignment, exploratory data analysis, feature engineering, chronological dataset splitting, model training, and performance evaluation.

Four forecasting approaches—Persistence Baseline, Linear Regression, ARIMA, and XGBoost—were compared using MAE, RMSE, and R².

**XGBoost achieved the best reported test-set performance**, with an MAE of 26.54, RMSE of 58.35, and R² of 0.8905. These results indicate that the model performed better than the other evaluated approaches on the selected test period.

The project also includes automated leakage checks and an interactive Streamlit dashboard for exploring solar generation patterns, understanding environmental relationships, and examining forecasting results.

Overall, the project demonstrates how machine learning and time-series analysis can be applied to short-term renewable-energy forecasting and provides a foundation for further work in solar power prediction, energy management, and intelligent renewable-energy systems.

---

⭐ If you find this project interesting, consider giving the repository a star!
