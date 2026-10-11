"""
evaluation.py
=============
Metrics, model comparison table and result graphs.

Run from the project root (after the model scripts):
    python src/evaluation.py

HOW IT WORKS
------------
Every model script saves its predictions to results/predictions/<model>.csv
with the columns: timestamp, split, y_true, y_pred.
This file reads ALL available prediction files and computes the metrics from
them, so every number in the comparison comes from real saved predictions.
A model whose script has not been run simply does not appear.

METRICS
-------
MAE  = average size of the error                (same unit as the power data)
RMSE = like MAE but big mistakes count extra    (same unit as the power data)
R2   = share of the variation explained         (1 = perfect, 0 = as good as
                                                 always predicting the average)

TWO ROW SETS
------------
"All rows"     : every row. About half of them are night rows where power is 0,
                 which makes the numbers look good.
"Daytime rows" : rows where POA irradiance at the forecast origin t is above
                 50 W/m2 (the same rule used in the EDA). This rule uses only a
                 value known at time t, so it does not depend on the answer.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
PRED_DIR = PROJECT_ROOT / "results" / "predictions"
GRAPH_DIR = PROJECT_ROOT / "results" / "graphs"
METRICS_FILE = PROJECT_ROOT / "results" / "metrics_all_splits.csv"
COMPARISON_FILE = PROJECT_ROOT / "results" / "model_comparison_test.csv"
LOCAL_TZ = "America/Los_Angeles"

DAYTIME_IRRADIANCE = 50.0  # W/m2
MIN_ROWS_PER_DAY = 90      # of 96 rows per day, used only to pick the graph window
MODEL_NAMES = {
    "baseline_persistence": "Baseline (Persistence)",
    "linear_regression": "Linear Regression",
    "xgboost": "XGBoost",
    "arima": "ARIMA",
}
COLORS = {
    "Baseline (Persistence)": "#7f7f7f",
    "Linear Regression": "#2ca02c",
    "XGBoost": "#d62728",
    "ARIMA": "#9467bd",
}


def regression_metrics(y_true, y_pred):
    """Return MAE, RMSE and R2 for two arrays."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def save_predictions(model_key, frame):
    """
    frame: DataFrame indexed by UTC timestamp with columns split, y_true, y_pred.
    Power cannot be negative, so predictions below 0 are set to 0 before saving.
    """
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    frame = frame[["split", "y_true", "y_pred"]].copy()
    n_negative = int((frame["y_pred"] < 0).sum())
    frame["y_pred"] = frame["y_pred"].clip(lower=0)
    frame.to_csv(PRED_DIR / f"{model_key}.csv")
    print(f"  saved results/predictions/{model_key}.csv "
          f"({len(frame)} rows; {n_negative} negative predictions set to 0)")


def load_predictions():
    """Read every available predictions file -> {display name: DataFrame}."""
    preds = {}
    for key, name in MODEL_NAMES.items():
        path = PRED_DIR / f"{key}.csv"
        if path.exists():
            preds[name] = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    return preds


def evaluate_all(preds, dataset):
    """Metrics for every model x split x row-set, on rows common to the models."""
    is_day = dataset["poa_irradiance"] > DAYTIME_IRRADIANCE
    rows = []
    for split in ["train", "validation", "test"]:
        have = {m: p[p["split"] == split] for m, p in preds.items() if (p["split"] == split).any()}
        if not have:
            continue
        # Use only timestamps that EVERY model (that has this split) predicted,
        # so all models are compared on exactly the same rows.
        common = None
        for p in have.values():
            common = p.index if common is None else common.intersection(p.index)
        for model, p in have.items():
            p = p.loc[common]
            for subset, mask in [("All rows", np.ones(len(p), dtype=bool)),
                                 ("Daytime rows", is_day.reindex(p.index).to_numpy())]:
                sel = p[mask]
                mae, rmse, r2 = regression_metrics(sel["y_true"], sel["y_pred"])
                rows.append({"model": model, "split": split, "row_set": subset,
                             "n_rows": len(sel), "MAE": mae, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows)


def pick_window(y_true, days=3):
    """
    Choose a representative period for the forecast graphs: the run of `days`
    consecutive well-covered days in the test set whose average daily total is
    closest to the average daily total of all well-covered test days.
    "Well-covered" = at least MIN_ROWS_PER_DAY of the 96 daily rows are present
    (a few rows per day were removed in feature filtering). Days are counted by
    the local date of the moment being forecast.
    """
    target_local = (y_true.index + pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ)
    dates = pd.Series(target_local.date, index=y_true.index)
    daily = y_true.groupby(dates).agg(["sum", "count"])
    totals = daily.loc[daily["count"] >= MIN_ROWS_PER_DAY, "sum"]
    if len(totals) < days:
        return y_true.index[0], y_true.index[min(len(y_true), days * 96) - 1]
    best, best_score = None, None
    ordered = sorted(totals.index)
    for i in range(len(ordered) - days + 1):
        window = ordered[i:i + days]
        if (window[-1] - window[0]).days != days - 1:
            continue  # days must follow each other
        score = abs(totals[window].mean() - totals.mean())
        if best_score is None or score < best_score:
            best, best_score = window, score
    if best is None:
        best = ordered[:days]
    in_window = dates.isin(best)
    return y_true.index[in_window][0], y_true.index[in_window][-1]


def make_graphs(preds, metrics):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    test_metrics = metrics[metrics["split"] == "test"]

    # ---- Model comparison bar chart ------------------------------------
    models = list(test_metrics["model"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        x = np.arange(len(models))
        for k, (subset, shade) in enumerate([("All rows", 1.0), ("Daytime rows", 0.55)]):
            vals = [test_metrics[(test_metrics["model"] == m) & (test_metrics["row_set"] == subset)][metric].iloc[0]
                    for m in models]
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, alpha=shade,
                   color=[COLORS[m] for m in models], edgecolor="black", linewidth=0.5,
                   label=subset)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=8)
        ax.set_title(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"))
        if metric == "R2":
            ax.axhline(0, color="black", linewidth=0.8)
    axes[0].legend(title="Solid = all rows\nLight = daytime", fontsize=7, title_fontsize=7)
    fig.suptitle("Model Performance on the Test Set")
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)

    # ---- Actual vs predicted in a representative period ------------------
    test_preds = {m: p[p["split"] == "test"] for m, p in preds.items()}
    common = None
    for p in test_preds.values():
        common = p.index if common is None else common.intersection(p.index)
    first = next(iter(test_preds.values())).loc[common]
    start, end = pick_window(first["y_true"])
    print(f"  representative period (origin times, UTC): {start} -> {end}")

    def to_local_target(index):  # plot predictions at the time they are FOR
        return (index + pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ).tz_localize(None)

    fig_all, ax_all = plt.subplots(figsize=(13, 4.8))
    shown_actual = False
    for model, p in test_preds.items():
        p = p.loc[common]
        w = p.loc[start:end]
        # Put the window on a complete 15-min grid so missing rows show as
        # breaks in the line instead of straight lines drawn across them.
        w = w.reindex(pd.date_range(start, end, freq="15min"))
        x = to_local_target(w.index)
        if not shown_actual:
            ax_all.plot(x, w["y_true"], color="black", linewidth=2.0, label="Actual")
            shown_actual = True
        ax_all.plot(x, w["y_pred"], color=COLORS[model], linewidth=1.3, label=model)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, w["y_true"], color="black", linewidth=1.8, label="Actual")
        ax.plot(x, w["y_pred"], color=COLORS[model], linewidth=1.5, label=f"Predicted ({model})")
        ax.set_title(f"Actual vs Predicted 1-Hour-Ahead Power - {model}")
        ax.set_xlabel("Time being forecast (local time)")
        ax.set_ylabel("AC power")
        ax.legend()
        fig.tight_layout()
        key = [k for k, v in MODEL_NAMES.items() if v == model][0]
        fig.savefig(GRAPH_DIR / f"actual_vs_predicted_{key}.png", dpi=150)
        plt.close(fig)
    ax_all.set_title("1-Hour-Ahead Forecasts vs Actual - Representative Test Period")
    ax_all.set_xlabel("Time being forecast (local time)")
    ax_all.set_ylabel("AC power")
    ax_all.legend(ncol=5, fontsize=8)
    fig_all.tight_layout()
    fig_all.savefig(GRAPH_DIR / "forecast_representative_period.png", dpi=150)
    plt.close(fig_all)


def main():
    dataset = pd.read_csv(DATASET_FILE, parse_dates=["timestamp"]).set_index("timestamp")
    preds = load_predictions()
    if not preds:
        print("No prediction files found in results/predictions/. Run the model scripts first.")
        return
    print("Models with saved predictions:", ", ".join(preds))
    metrics = evaluate_all(preds, dataset)
    metrics.to_csv(METRICS_FILE, index=False)

    pd.set_option("display.width", 200)
    for split in ["validation", "test"]:
        part = metrics[metrics["split"] == split]
        if part.empty:
            continue
        print(f"\n===== {split.upper()} SET =====")
        for subset in ["All rows", "Daytime rows"]:
            t = part[part["row_set"] == subset][["model", "n_rows", "MAE", "RMSE", "R2"]]
            print(f"\n{subset}")
            print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    test = metrics[metrics["split"] == "test"]
    test.to_csv(COMPARISON_FILE, index=False)
    print("\n===== BEST MODEL ON THE TEST SET (from the numbers above) =====")
    for subset in ["All rows", "Daytime rows"]:
        part = test[test["row_set"] == subset]
        print(f"{subset}: lowest MAE = {part.loc[part['MAE'].idxmin(), 'model']}; "
              f"lowest RMSE = {part.loc[part['RMSE'].idxmin(), 'model']}; "
              f"highest R2 = {part.loc[part['R2'].idxmax(), 'model']}")

    print("\nGraphs:")
    make_graphs(preds, metrics)
    print(f"\nSaved {METRICS_FILE.name}, {COMPARISON_FILE.name} and graphs in results/graphs/")


if __name__ == "__main__":
    main()
