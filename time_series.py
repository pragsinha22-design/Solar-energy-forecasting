"""
time_series.py
==============
PHASE 7 - ARIMA forecasting of AC power (statsmodels).

Run from the project root:
    python src/time_series.py

Output: results/predictions/arima.csv, results/arima_order_selection.csv,
        results/arima_summary.txt   (then run src/evaluation.py)

WHAT IS ARIMA?
--------------
ARIMA(p, d, q) predicts a series from its OWN past only (here: AC power).
    p = AutoRegressive order : how many past VALUES are used
                               (p=2 -> uses power 15 and 30 minutes ago)
    d = Differencing order   : how many times we difference the series
                               (d=1 -> model changes in power, not power itself)
                               d=0 if the series is already stationary.
    q = Moving-average order : how many past forecast ERRORS are used
Unlike the regression models, plain ARIMA does NOT use weather features.

HOW TRAINING AND TESTING ARE HANDLED
------------------------------------
1. The model parameters are estimated ONCE, using only power values from before
   the validation period starts (the same data the regression models train on).
2. We then walk through the TEST period in time order. At each forecast origin t
   the model sees power up to and including t, and predicts t+15, t+30, t+45
   and t+60 minutes. We keep the t+60 minute value (1 hour ahead) and compare
   it with the real power. Then the real value at the next origin is revealed
   and we repeat. The parameters are NOT re-estimated, so nothing from the test
   period influences them.
3. The forecast origins are exactly the test rows used by the other models, so
   every model is scored on the same rows.
Missing observations (the two clock-change gaps) are left as NaN; the
state-space ARIMA in statsmodels handles missing values natively.
"""

import time
import warnings
import numpy as np
import pandas as pd

from features import PROJECT_ROOT, MERGED_FILE, OUT_FILE, TARGET_COLUMN, STEPS_PER_HOUR
from evaluation import save_predictions, regression_metrics

# Candidate (p, q) pairs. Small on purpose: this is a college project and the
# training series has ~45,000 points. d is chosen from a stationarity test.
CANDIDATE_PQ = [(1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    warnings.filterwarnings("ignore")  # hide convergence chatter, we print what matters

    # ------------------------------------------------------------------
    # Load: full 15-minute power series (with NaN where the data had gaps)
    # and the list of test forecast origins.
    # ------------------------------------------------------------------
    merged = pd.read_csv(MERGED_FILE, parse_dates=["timestamp"]).set_index("timestamp")
    dataset = pd.read_csv(OUT_FILE, parse_dates=["timestamp"]).set_index("timestamp")

    grid = pd.date_range(merged.index.min(), merged.index.max(), freq="15min")
    power = merged["ac_power"].reindex(grid).to_numpy(dtype=float)  # plain numbers, one per 15 min

    val_start = dataset.index[dataset["split"] == "validation"].min()
    n_train = int((grid < val_start).sum())          # everything before validation starts
    train_values = power[:n_train]
    test_origins = dataset.index[dataset["split"] == "test"]
    test_positions = grid.get_indexer(test_origins)  # where each origin sits in the grid
    assert (test_positions >= n_train).all() and (test_positions >= 0).all()

    print(f"ARIMA training series : {n_train} points "
          f"({grid[0]} -> {grid[n_train - 1]}), NaN count = {int(np.isnan(train_values).sum())}")
    print(f"Test forecast origins : {len(test_origins)} "
          f"({test_origins.min()} -> {test_origins.max()})")

    # ------------------------------------------------------------------
    # Step 1: stationarity test -> choose d
    # Augmented Dickey-Fuller: null hypothesis = "the series is NOT stationary".
    # p-value < 0.05 -> reject it -> treat as stationary -> d = 0.
    # ------------------------------------------------------------------
    adf_stat, adf_p = adfuller(train_values[~np.isnan(train_values)])[:2]
    d = 0 if adf_p < 0.05 else 1
    print(f"\nADF test on training power: statistic={adf_stat:.3f}, p-value={adf_p:.4g} "
          f"-> {'stationary, d = 0' if d == 0 else 'not stationary, d = 1'}")

    # ------------------------------------------------------------------
    # Step 2: choose (p, q) with the lowest AIC on the TRAINING data only.
    # AIC = fit quality with a penalty for extra parameters (lower is better).
    # ------------------------------------------------------------------
    print("\nFitting candidate models on the training series (this can take a few minutes):")
    rows, best = [], None
    for p, q in CANDIDATE_PQ:
        start = time.time()
        try:
            fitted = ARIMA(train_values, order=(p, d, q)).fit()
            aic = float(fitted.aic)
        except Exception as error:  # a candidate may fail to converge; skip it
            print(f"  ARIMA({p},{d},{q}) failed: {error}")
            continue
        seconds = time.time() - start
        rows.append({"order": f"({p},{d},{q})", "AIC": aic, "fit_seconds": round(seconds, 1)})
        print(f"  ARIMA({p},{d},{q})  AIC = {aic:,.1f}   [{seconds:.0f}s]")
        if best is None or aic < best[0]:
            best = (aic, (p, d, q), fitted)
    if best is None:
        raise RuntimeError("No ARIMA candidate could be fitted.")
    _, order, fitted = best
    print(f"\nSelected order: ARIMA{order}")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "arima_order_selection.csv", index=False)
    (RESULTS_DIR / "arima_summary.txt").write_text(str(fitted.summary()), encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 3: rolling 1-hour-ahead forecasts through the test period.
    # 'results' always holds the model state at the latest revealed time.
    # results.extend(new_values) feeds in new observations WITHOUT changing the
    # estimated parameters. results.forecast(4) predicts the next 4 steps.
    # ------------------------------------------------------------------
    print(f"\nRolling forecasts for {len(test_positions)} test origins "
          f"(parameters stay fixed)...")
    results = fitted
    last_seen = n_train - 1          # grid position of the latest observation the model has seen
    predictions = np.zeros(len(test_positions))
    start = time.time()
    for k, position in enumerate(test_positions):
        if position > last_seen:     # reveal the real values up to the origin t (inclusive)
            results = results.extend(power[last_seen + 1: position + 1])
            last_seen = position
        forecast = np.asarray(results.forecast(steps=STEPS_PER_HOUR))
        predictions[k] = forecast[-1]  # the value 4 steps = 1 hour after t
        if (k + 1) % 1000 == 0:
            print(f"  {k + 1}/{len(test_positions)} origins done "
                  f"({time.time() - start:.0f}s)")

    frame = pd.DataFrame({
        "split": "test",
        "y_true": dataset.loc[test_origins, TARGET_COLUMN],
        "y_pred": predictions,
    }, index=test_origins)
    save_predictions("arima", frame)

    clipped = frame["y_pred"].clip(lower=0)
    mae, rmse, r2 = regression_metrics(frame["y_true"], clipped)
    print(f"\nARIMA{order} on the test set (all rows): MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")
    print("Run  python src/evaluation.py  for the full comparison.")


if __name__ == "__main__":
    main()
