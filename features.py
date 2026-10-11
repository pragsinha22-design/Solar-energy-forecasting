"""
features.py
===========
PHASE 3 - Feature engineering, 1-hour-ahead target, filtering, chronological split.

Run from the project root:
    python src/features.py

Reads  : data/processed/merged_15min.csv
         data/processed/filled_flags.csv
Writes : data/processed/model_dataset.csv   (features + target + split label)
         results/features_report.txt

THE FORECASTING PROBLEM
-----------------------
"Standing at time t, using only what is known at time t, predict the AC power
 at time t + 1 hour."   The data has one row every 15 minutes, so
 t + 1 hour is exactly 4 rows later.  (Row number would be dangerous if rows
 were missing, so we put the data on a complete 15-minute time grid first.)
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_FILE = PROJECT_ROOT / "data" / "processed" / "merged_15min.csv"
FLAGS_FILE = PROJECT_ROOT / "data" / "processed" / "filled_flags.csv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "model_dataset.csv"
REPORT_FILE = PROJECT_ROOT / "results" / "features_report.txt"

LOCAL_TZ = "America/Los_Angeles"
STEPS_PER_HOUR = 4  # 4 rows x 15 minutes = 1 hour

TARGET_COLUMN = "power_1h_ahead"
FEATURE_COLUMNS = [
    # --- power at/before time t -------------------------------------------
    "ac_power",           # power at time t (the "now" value)
    "power_lag_1",        # power 1 step  = 15 minutes before t
    "power_lag_2",        # power 2 steps = 30 minutes before t
    "power_lag_4",        # power 4 steps = 60 minutes before t
    # --- weather at time t -------------------------------------------------
    "ambient_temperature",
    "wind_speed",
    "wind_direction",
    "poa_irradiance",
    # --- calendar features (local clock time of t) -------------------------
    "hour",
    "day",
    "month",
    "day_of_week",
    # --- clock time of the moment being forecast (t + 1 hour) ---------------
    "target_hour_sin",
    "target_hour_cos",
]
BASE_COLUMNS = ["ac_power", "ambient_temperature", "wind_speed", "wind_direction", "poa_irradiance"]
WEATHER_COLUMNS = ["ambient_temperature", "wind_speed", "wind_direction", "poa_irradiance"]


def build_features(raw):
    """
    Turn the merged data into a feature table.

    raw : DataFrame indexed by UTC timestamp with the BASE_COLUMNS.
    Every feature in row t is computed ONLY from raw data at times <= t,
    except the target, which by definition looks 1 hour ahead.
    """
    # 1) Put the data on a COMPLETE 15-minute grid. Where the original data had
    #    a gap, a row with NaN appears. This guarantees shift(4) == exactly 1 hour.
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="15min")
    assert raw.index.isin(grid).all(), "Some timestamps are not on the 15-minute grid"
    data = raw[BASE_COLUMNS].reindex(grid)
    data.index.name = "timestamp"

    out = pd.DataFrame(index=data.index)

    # 2) Power now and lagged power.  shift(k) moves values DOWN by k rows, so
    #    row t receives the value from row t-k  (the PAST). Never shift(-k) here.
    out["ac_power"] = data["ac_power"]
    out["power_lag_1"] = data["ac_power"].shift(1)
    out["power_lag_2"] = data["ac_power"].shift(2)
    out["power_lag_4"] = data["ac_power"].shift(4)

    # 3) Weather measured at time t (current values only, no future values)
    for col in WEATHER_COLUMNS:
        out[col] = data[col]

    # 4) Calendar features from the LOCAL clock time of t
    local = data.index.tz_convert(LOCAL_TZ)
    out["hour"] = local.hour
    out["day"] = local.day
    out["month"] = local.month
    out["day_of_week"] = local.dayofweek  # Monday=0 ... Sunday=6

    # 5) Clock time of the moment we are forecasting (t + 1 hour). A calendar is
    #    known in advance, so this is NOT leakage. We use sin/cos so that 23:xx
    #    and 00:xx are "close" to each other (hours form a circle, not a line).
    target_local = (data.index + pd.Timedelta(hours=1)).tz_convert(LOCAL_TZ)
    fractional_hour = target_local.hour + target_local.minute / 60
    out["target_hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24)
    out["target_hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24)

    # 6) THE TARGET: power one hour (4 steps) AFTER t.  shift(-4) pulls values
    #    UP from the future. It is only ever used as the answer, never as input.
    out[TARGET_COLUMN] = data["ac_power"].shift(-STEPS_PER_HOUR)

    return out


def assign_split(index, train_frac=0.70, val_frac=0.15):
    """
    Chronological split: earliest 70% train, next 15% validation, last 15% test.

    Purging: a training row at time t has its answer at t + 1 hour. For the last
    rows of the training block that answer would fall INSIDE the validation
    period. We label those rows 'purged' and do not use them, so no training
    answer overlaps with the next block. Same at the validation/test boundary.
    """
    n = len(index)
    val_start = index[int(n * train_frac)]
    test_start = index[int(n * (train_frac + val_frac))]
    horizon = pd.Timedelta(hours=1)

    split = pd.Series("test", index=index)
    split[index < test_start] = "validation"
    split[index < val_start] = "train"
    split[(split == "train") & (index + horizon >= val_start)] = "purged"
    split[(split == "validation") & (index + horizon >= test_start)] = "purged"
    return split


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(str(text))

    # ------------------------------------------------------------------
    # Load merged data (kept exactly as produced by preprocessing.py)
    # ------------------------------------------------------------------
    raw = pd.read_csv(MERGED_FILE, parse_dates=["timestamp"]).set_index("timestamp")
    flags = pd.read_csv(FLAGS_FILE, parse_dates=["timestamp"]).set_index("timestamp")
    log(f"Merged data loaded: {raw.shape}")

    # ------------------------------------------------------------------
    # Build features + target on the complete 15-minute grid
    # ------------------------------------------------------------------
    feats = build_features(raw)
    n_inserted = len(feats) - len(raw)
    log(f"Complete 15-min UTC grid: {len(feats)} rows "
        f"({n_inserted} empty rows inserted where the original data had gaps)")

    # ------------------------------------------------------------------
    # Row filtering: figure out WHY each unusable row is unusable
    # ------------------------------------------------------------------
    was_filled = flags.reindex(feats.index).fillna(False).astype(bool).any(axis=1)
    missing_feature = feats[FEATURE_COLUMNS].isna().any(axis=1)
    missing_target = feats[TARGET_COLUMN].isna()
    missing_weather = feats[WEATHER_COLUMNS].isna().any(axis=1)
    missing_power_inputs = feats[["ac_power", "power_lag_1", "power_lag_2", "power_lag_4"]].isna().any(axis=1)

    drop = missing_feature | missing_target | was_filled
    log()
    log("Rows removed from modelling (a row can have more than one reason):")
    log(f"  weather value still missing (gap > 1 hour or end of file): {int(missing_weather.sum())}")
    log(f"  weather value was interpolated in preprocessing:           {int(was_filled.sum())}")
    log(f"  power / lag value missing (start of data or around gaps):  {int(missing_power_inputs.sum())}")
    log(f"  target (power 1h ahead) missing:                           {int(missing_target.sum())}")
    log(f"  TOTAL rows removed (union of the reasons):                 {int(drop.sum())}")

    data = feats[~drop].copy()
    log(f"\nDataset after filtering: {data.shape[0]} rows x {data.shape[1]} columns "
        f"({data.shape[1] - 1} features + 1 target)  "
        f"[{drop.sum() / len(feats) * 100:.2f}% of the grid removed]")

    # ------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------
    data["split"] = assign_split(data.index)
    data["timestamp_local"] = data.index.tz_convert(LOCAL_TZ).tz_localize(None)

    log("\nChronological split (no shuffling):")
    log(f"{'split':<12}{'rows':>8}   {'first (local)':<22}{'last (local)':<22}"
        f"{'mean target':>12}{'% zero':>8}")
    for name in ["train", "validation", "test", "purged"]:
        part = data[data["split"] == name]
        if part.empty:
            continue
        log(f"{name:<12}{len(part):>8}   {str(part['timestamp_local'].min()):<22}"
            f"{str(part['timestamp_local'].max()):<22}"
            f"{part[TARGET_COLUMN].mean():>12.1f}{(part[TARGET_COLUMN] == 0).mean() * 100:>7.1f}%")
    used = data["split"] != "purged"
    log(f"Rows used for modelling (train+validation+test): {int(used.sum())}; "
        f"purged boundary rows: {int((~used).sum())}")

    # Save
    data = data[["timestamp_local", "split"] + FEATURE_COLUMNS + [TARGET_COLUMN]]
    data.to_csv(OUT_FILE)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nSaved {OUT_FILE}")


if __name__ == "__main__":
    main()
