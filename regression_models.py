"""
regression_models.py
====================
PHASES 4, 5, 6 - Baseline, Linear Regression and XGBoost.

Run from the project root:
    python src/regression_models.py                      # all three
    python src/regression_models.py --models baseline linear
    python src/regression_models.py --models xgboost

All models use the SAME chronological split created in features.py:
    train (earliest 70%)  ->  validation (next 15%)  ->  test (last 15%)
The test set is never used for training or tuning.

Output: results/predictions/<model>.csv  (then run src/evaluation.py)
"""

import argparse
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN, PROJECT_ROOT, OUT_FILE
from evaluation import save_predictions, regression_metrics


def load_splits():
    """Read the model dataset and cut it into the three chronological blocks."""
    data = pd.read_csv(OUT_FILE, parse_dates=["timestamp"]).set_index("timestamp")
    splits = {name: data[data["split"] == name] for name in ["train", "validation", "test"]}
    for name, part in splits.items():
        print(f"  {name:<11}{len(part):>7} rows   {part.index.min()} -> {part.index.max()}")
    return splits


def make_prediction_frame(splits, predict_function):
    """Predict every split and collect (split, y_true, y_pred) in one table."""
    pieces = []
    for name, part in splits.items():
        piece = pd.DataFrame({
            "split": name,
            "y_true": part[TARGET_COLUMN],
            "y_pred": predict_function(part),
        }, index=part.index)
        pieces.append(piece)
    return pd.concat(pieces)


def quick_report(frame):
    """Print MAE/RMSE/R2 per split (full tables are produced by evaluation.py)."""
    clipped = frame.copy()
    clipped["y_pred"] = clipped["y_pred"].clip(lower=0)
    for name in ["train", "validation", "test"]:
        part = clipped[clipped["split"] == name]
        mae, rmse, r2 = regression_metrics(part["y_true"], part["y_pred"])
        print(f"    {name:<11} MAE={mae:8.3f}  RMSE={rmse:8.3f}  R2={r2:7.4f}")


# ----------------------------------------------------------------------
# PHASE 4 - BASELINE: persistence ("nothing changes")
# ----------------------------------------------------------------------
def run_baseline(splits):
    """
    Persistence forecast: "power in one hour = power right now".
    It needs no training. Any real model must beat it to be worth using,
    so it is the minimum standard every other model is compared against.
    """
    print("\n[Baseline: persistence]")
    frame = make_prediction_frame(splits, lambda part: part["ac_power"])
    quick_report(frame)
    save_predictions("baseline_persistence", frame)


# ----------------------------------------------------------------------
# PHASE 5 - LINEAR REGRESSION
# ----------------------------------------------------------------------
def run_linear_regression(splits):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print("\n[Linear Regression]")
    # The scaler is learned from the TRAINING block only (inside the pipeline),
    # then applied unchanged to validation and test. This avoids leakage.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression()),
    ])
    model.fit(splits["train"][FEATURE_COLUMNS], splits["train"][TARGET_COLUMN])

    frame = make_prediction_frame(splits, lambda part: model.predict(part[FEATURE_COLUMNS]))
    quick_report(frame)
    save_predictions("linear_regression", frame)

    # Coefficients are on standardised features, so their sizes are comparable.
    coefs = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": model.named_steps["regression"].coef_,
    })
    coefs["abs_coefficient"] = coefs["coefficient"].abs()
    coefs = coefs.sort_values("abs_coefficient", ascending=False)
    coefs.to_csv(PROJECT_ROOT / "results" / "linear_regression_coefficients.csv", index=False)
    print("    Largest standardised coefficients:")
    print(coefs.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ----------------------------------------------------------------------
# PHASE 6 - XGBOOST
# ----------------------------------------------------------------------
def run_xgboost(splits):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("\n[XGBoost] SKIPPED: xgboost is not installed. Run: pip install xgboost")
        return

    print("\n[XGBoost]")
    # Parameters explained (kept simple on purpose):
    #   n_estimators=500      maximum number of small decision trees
    #   learning_rate=0.05    how much each new tree corrects the previous ones
    #   max_depth=6           how complex each tree can be
    #   subsample=0.8         each tree sees a random 80% of the rows
    #   colsample_bytree=0.8  each tree sees a random 80% of the features
    #   early_stopping_rounds=30  stop adding trees when the VALIDATION error
    #                             has not improved for 30 rounds
    # The validation block is used only to decide when to stop. The test block
    # is never seen during training.
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=42,
    )
    model.fit(
        splits["train"][FEATURE_COLUMNS], splits["train"][TARGET_COLUMN],
        eval_set=[(splits["validation"][FEATURE_COLUMNS], splits["validation"][TARGET_COLUMN])],
        verbose=False,
    )
    best_round = getattr(model, "best_iteration", None)
    print(f"    trees used (best round on validation): {best_round}")

    frame = make_prediction_frame(splits, lambda part: model.predict(part[FEATURE_COLUMNS]))
    quick_report(frame)
    save_predictions("xgboost", frame)

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(PROJECT_ROOT / "results" / "xgboost_feature_importance.csv", index=False)
    print("    Top feature importances:")
    print(importance.head(5).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["baseline", "linear", "xgboost"],
                        choices=["baseline", "linear", "xgboost"])
    args = parser.parse_args()

    print("Loading the chronological splits:")
    splits = load_splits()
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)

    if "baseline" in args.models:
        run_baseline(splits)
    if "linear" in args.models:
        run_linear_regression(splits)
    if "xgboost" in args.models:
        run_xgboost(splits)


if __name__ == "__main__":
    main()
