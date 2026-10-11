"""
Solar Energy Forecasting dashboard.

Run from the project root:
    streamlit run dashboard/app.py

The dashboard only READS files created by the pipeline (python main.py):
    data/processed/merged_15min.csv
    data/processed/model_dataset.csv
    results/predictions/<model>.csv
    results/metrics_all_splits.csv
It does not retrain anything, and it never invents numbers: a model that has not
been run is simply not shown.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MERGED_FILE = ROOT / "data" / "processed" / "merged_15min.csv"
DATASET_FILE = ROOT / "data" / "processed" / "model_dataset.csv"
PRED_DIR = ROOT / "results" / "predictions"
METRICS_FILE = ROOT / "results" / "metrics_all_splits.csv"
LOCAL_TZ = "America/Los_Angeles"

MODELS = {
    "baseline_persistence": "Baseline (Persistence)",
    "linear_regression": "Linear Regression",
    "xgboost": "XGBoost",
    "arima": "ARIMA",
}

st.set_page_config(page_title="Solar Energy Forecasting", page_icon="☀️", layout="wide")


# ----------------------------------------------------------------------
# Loading helpers (cached so the page stays fast)
# ----------------------------------------------------------------------
@st.cache_data
def load_merged():
    df = pd.read_csv(MERGED_FILE, parse_dates=["timestamp_local"])
    return df.set_index("timestamp_local")


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_FILE, parse_dates=["timestamp"]).set_index("timestamp")


@st.cache_data
def load_predictions():
    """{model name: test-set predictions} for every model that has been run."""
    found = {}
    for key, name in MODELS.items():
        path = PRED_DIR / f"{key}.csv"
        if path.exists():
            p = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
            found[name] = p[p["split"] == "test"]
    return found


@st.cache_data
def load_metrics():
    if METRICS_FILE.exists():
        return pd.read_csv(METRICS_FILE)
    return None


def target_local_time(index):
    """A prediction made at time t is FOR t + 1 hour. Return that time in local clock time."""
    return (index + pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ).tz_localize(None)


# ----------------------------------------------------------------------
# Check that the pipeline has been run
# ----------------------------------------------------------------------
if not (MERGED_FILE.exists() and DATASET_FILE.exists()):
    st.title("☀️ Short-Term Solar Energy Generation Forecasting")
    st.error("Processed data not found. Run  `python main.py`  from the project folder first.")
    st.stop()

merged = load_merged()
dataset = load_dataset()
predictions = load_predictions()
metrics = load_metrics()

# ----------------------------------------------------------------------
# 1. Title
# ----------------------------------------------------------------------
st.title("☀️ Short-Term Solar Energy Generation Forecasting")
st.caption("Predicting solar PV AC power output 1 hour ahead from historical power and weather data "
           "(PVDAQ / OEDI system 2107, 15-minute data).")

# ----------------------------------------------------------------------
# 2. Dataset summary
# ----------------------------------------------------------------------
st.header("1. Dataset summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (15-min)", f"{len(merged):,}")
c2.metric("First record (local)", str(merged.index.min().date()))
c3.metric("Last record (local)", str(merged.index.max().date()))
split_counts = dataset["split"].value_counts()
c4.metric("Test rows", f"{int(split_counts.get('test', 0)):,}")

stats_cols = ["ac_power", "ambient_temperature", "wind_speed", "wind_direction", "poa_irradiance"]
st.dataframe(merged[stats_cols].describe().round(2).T)
st.caption("Meter timestamps were converted from local Pacific time to UTC to match the weather files. "
           "Temperature and wind units are shown as recorded in the files (not stated there).")

# ----------------------------------------------------------------------
# 3. Solar generation graph
# ----------------------------------------------------------------------
st.header("2. Solar generation")
first_day, last_day = merged.index.min().date(), merged.index.max().date()
default_start = max(first_day, (merged.index.max() - pd.Timedelta(days=7)).date())
chosen = st.date_input("Date range (local time)", value=(default_start, last_day),
                       min_value=first_day, max_value=last_day)
if isinstance(chosen, (tuple, list)):
    start_day, end_day = (chosen[0], chosen[-1]) if len(chosen) > 0 else (default_start, last_day)
else:
    start_day, end_day = chosen, chosen
window = merged.loc[str(start_day): str(end_day) + " 23:59:59"]

resolution = st.radio("Resolution", ["15 minutes (raw)", "Hourly mean", "Daily mean"], horizontal=True)
rule = {"15 minutes (raw)": None, "Hourly mean": "h", "Daily mean": "D"}[resolution]
view = window if rule is None else window[stats_cols].resample(rule).mean()
st.line_chart(view["ac_power"], y_label="AC power")

# ----------------------------------------------------------------------
# 4. Weather / irradiance
# ----------------------------------------------------------------------
st.header("3. Weather and irradiance")
weather_choice = st.selectbox("Variable", ["poa_irradiance", "ambient_temperature", "wind_speed"])
st.line_chart(view[weather_choice], y_label=weather_choice)

left, right = st.columns(2)
with left:
    st.subheader("Average power by hour of day")
    by_hour = merged.groupby(merged.index.hour)["ac_power"].mean()
    by_hour.index.name = "hour (local)"
    st.bar_chart(by_hour)
with right:
    st.subheader("Correlation with AC power")
    corr = merged[stats_cols].corr()["ac_power"].drop("ac_power").round(3)
    st.dataframe(corr.rename("correlation"))
    st.caption("Over all rows, including night. Irradiance is by far the strongest driver.")

# ----------------------------------------------------------------------
# 5. Models available + model selection
# ----------------------------------------------------------------------
st.header("4. Forecasts")
if not predictions:
    st.warning("No model predictions found. Run `python main.py` to create them.")
    st.stop()

missing = [name for name in MODELS.values() if name not in predictions]
if missing:
    st.info("Not run yet (no saved predictions): " + ", ".join(missing) +
            ".  Run `python main.py` to add them.")

model_name = st.sidebar.selectbox("Model", list(predictions), index=min(1, len(predictions) - 1))
show_baseline = st.sidebar.checkbox("Also show baseline (persistence)", value=True)
st.sidebar.caption("All forecasts are for the test period, which the models never saw in training.")

pred = predictions[model_name].copy()
pred["target_time"] = target_local_time(pred.index)
pred["date"] = pred["target_time"].dt.date

# ----------------------------------------------------------------------
# 6. Forecast output + actual vs predicted
# ----------------------------------------------------------------------
days = sorted(pred["date"].unique())
day = st.selectbox("Day being forecast (local date)", days, index=len(days) // 2)
day_rows = pred[pred["date"] == day].sort_values("target_time")

# Actual vs predicted chart (both plotted at the time the forecast is FOR)
chart = pd.DataFrame({"Actual": day_rows["y_true"].to_numpy(),
                      model_name: day_rows["y_pred"].to_numpy()},
                     index=day_rows["target_time"])
if show_baseline and model_name != "Baseline (Persistence)" and "Baseline (Persistence)" in predictions:
    base = predictions["Baseline (Persistence)"]
    base = base.reindex(day_rows.index)
    chart["Baseline (Persistence)"] = base["y_pred"].to_numpy()
st.subheader(f"Actual vs predicted - {model_name}")
st.line_chart(chart, y_label="AC power")
st.caption("Each prediction was made 1 hour earlier, using only data available at that earlier time.")

# Single forecast output
st.subheader("Forecast output")
raw_labels = (day_rows["target_time"] - pd.Timedelta(hours=1)).dt.strftime("%H:%M").tolist()
origin_labels, seen = [], set()
for label in raw_labels:  # clocks repeat one hour in November, so keep labels unique
    origin_labels.append(label + " (repeated hour)" if label in seen else label)
    seen.add(label)
if len(origin_labels) > 0:
    nearest = int((day_rows["target_time"].dt.hour + day_rows["target_time"].dt.minute / 60 - 12).abs().argmin())
    chosen_origin = st.select_slider("Forecast made at (local time)", options=origin_labels,
                                     value=origin_labels[nearest])
    row = day_rows.iloc[origin_labels.index(chosen_origin)]
    o1, o2, o3 = st.columns(3)
    o1.metric("Forecast for", row["target_time"].strftime("%Y-%m-%d %H:%M"))
    o2.metric("Predicted AC power", f"{row['y_pred']:.1f}")
    o3.metric("Actual AC power", f"{row['y_true']:.1f}",
              delta=f"prediction error {row['y_pred'] - row['y_true']:+.1f}", delta_color="off")
with st.expander("Show the table for this day"):
    table = day_rows[["target_time", "y_true", "y_pred"]].rename(
        columns={"target_time": "forecast for (local)", "y_true": "actual", "y_pred": "predicted"})
    table["error"] = table["predicted"] - table["actual"]
    table[["actual", "predicted", "error"]] = table[["actual", "predicted", "error"]].round(2)
    st.dataframe(table, hide_index=True)

# ----------------------------------------------------------------------
# 7. Metrics for the selected model + model comparison
# ----------------------------------------------------------------------
st.header("5. Model performance (test set)")
if metrics is None:
    st.warning("Metrics file not found. Run `python src/evaluation.py`.")
else:
    test_metrics = metrics[metrics["split"] == "test"]

    st.subheader(f"{model_name}")
    for subset in ["All rows", "Daytime rows"]:
        part = test_metrics[(test_metrics["model"] == model_name) & (test_metrics["row_set"] == subset)]
        if part.empty:
            continue
        r = part.iloc[0]
        st.markdown(f"**{subset}**  ({int(r['n_rows']):,} rows)")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{r['MAE']:.2f}")
        m2.metric("RMSE", f"{r['RMSE']:.2f}")
        m3.metric("R²", f"{r['R2']:.4f}")

    st.subheader("Model comparison")
    row_set = st.radio("Rows to evaluate", ["All rows", "Daytime rows"], horizontal=True)
    comp = test_metrics[test_metrics["row_set"] == row_set].set_index("model")
    comp = comp[["n_rows", "MAE", "RMSE", "R2"]].sort_values("RMSE")
    st.dataframe(comp.round(4))
    best = comp["RMSE"].idxmin()
    st.success(f"Lowest RMSE on the test set ({row_set.lower()}): **{best}** "
               f"({comp.loc[best, 'RMSE']:.2f}).")
    b1, b2 = st.columns(2)
    b1.bar_chart(comp[["MAE", "RMSE"]])
    b2.bar_chart(comp[["R2"]])
    st.caption("Daytime rows = POA irradiance above 50 W/m² at the time the forecast is made. "
               "The test period covers autumn and winter 2025, so scores reflect those conditions.")

    with st.expander("Validation-set metrics"):
        val = metrics[(metrics["split"] == "validation") & (metrics["row_set"] == row_set)]
        st.dataframe(val.drop(columns=["split"]).round(4), hide_index=True)

st.divider()
st.caption("College project: Short-Term Solar Energy Generation Forecasting Using Historical Power and "
           "Weather Data. All numbers shown are read from files produced by the project pipeline.")
